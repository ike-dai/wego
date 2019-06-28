package corpus

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"math/rand"
	"strings"

	lingo "github.com/chewxy/lingo/corpus"
	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"

	"github.com/ynqa/wego/co"
	"github.com/ynqa/wego/model"
	"github.com/ynqa/wego/node"
	"github.com/ynqa/wego/timer"
)

type WegoCorpus struct {
	*lingo.Corpus
	// Document stores a list of word indexes for model.Mode == Memory
	Document []int
}

func NewWegoCorpus(mode model.Mode) *WegoCorpus {
	c, _ := lingo.Construct()
	return &WegoCorpus{
		Corpus:   c,
		Document: make([]int, 0),
	}
}

func (c *WegoCorpus) Parse(r io.Reader, toLower bool, minCount int, batchSize int, f func()) error {
	scanner := bufio.NewScanner(r)
	scanner.Split(bufio.ScanWords)
	var i int
	buff := make([]int, batchSize)
	for scanner.Scan() {
		word := scanner.Text()
		if toLower {
			word = strings.ToLower(word)
		}
		buff[i], _ = c.Id(word)
		if i%batchSize == 0 {
			go f()
			buff = make([]int, batchSize)
		}
		i++
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.Wrap(err, "Unable to complete scanning")
	}
	return nil
}

func (c *WegoCorpus) BuildBatch(f io.Reader, toLower bool, minCount int, batchSize int, verbose bool) error {
	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)

	var t *timer.Timer
	if verbose {
		t = timer.NewTimer()
	}

	var i int
	for scanner.Scan() {
		word := scanner.Text()
		if toLower {
			word = strings.ToLower(word)
		}
		c.Add(word)
		if verbose && i%batchSize == 0 {
			fmt.Printf("Read %d words %v\r", i, t.AllElapsed())
		}
		i++
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.Wrap(err, "Unable to complete scanning")
	}
	if verbose {
		fmt.Printf("Read %d words %v\r\n", i, t.AllElapsed())
	}
	return nil
}

func (c *WegoCorpus) Build(f io.Reader, toLower bool, minCount int, batchSize int, verbose bool) error {
	fullDoc := make([]int, 0)
	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)

	var t *timer.Timer
	if verbose {
		t = timer.NewTimer()
	}
	var i int
	for scanner.Scan() {
		word := scanner.Text()
		if toLower {
			word = strings.ToLower(word)
		}
		// TODO: delete words less than minCount in Corpus.
		c.Add(word)
		wordID, _ := c.Id(word)
		fullDoc = append(fullDoc, wordID)
		if verbose && i%batchSize == 0 {
			fmt.Printf("Read %d words %v\r", i, t.AllElapsed())
		}
		i++
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.Wrap(err, "Unable to complete scanning")
	}
	if verbose {
		fmt.Printf("Read %d words %v\r\n", i, t.AllElapsed())
	}
	for _, d := range fullDoc {
		if c.IDFreq(d) > minCount {
			c.Document = append(c.Document, d)
		}
	}
	if verbose {
		fmt.Printf("Filter words less than minCount=%d > documentSize=%d\n", minCount, len(c.Document))
	}
	return nil
}

// HuffmanTree builds word nodes map.
func (c *WegoCorpus) HuffmanTree(dimension int) (map[int]*node.Node, error) {
	ns := make(node.Nodes, 0, c.Size())
	nm := make(map[int]*node.Node)
	for i := 0; i < c.Size(); i++ {
		n := new(node.Node)
		n.Value = c.IDFreq(i)
		nm[i] = n
		ns = append(ns, n)
	}
	if err := ns.Build(dimension); err != nil {
		return nil, err
	}
	return nm, nil
}

// RelationType is a list of types for strength relations between co-occurrence words.
type RelationType int

const (
	PPMI RelationType = iota
	PMI
	CO
	LOGCO
)

// String describes relation type name.
func (r RelationType) String() string {
	switch r {
	case PPMI:
		return "ppmi"
	case PMI:
		return "pmi"
	case CO:
		return "co"
	case LOGCO:
		return "logco"
	default:
		return "unknown"
	}
}

// CountType is a list of types to count co-occurences.
type CountType int

const (
	INCREMENT CountType = iota
	// DISTANCE weights values for co-occurrence times.
	DISTANCE
)

func (c *WegoCorpus) cooccurrence(mode model.Mode, window int, typ CountType, verbose bool) (map[uint64]float64, error) {
	switch mode {
	case model.Memory:
		documentSize := len(c.Document)

		var progress *pb.ProgressBar
		if verbose {
			fmt.Println("Scan corpus for cooccurrences")
			progress = pb.New(documentSize).SetWidth(80)
			defer progress.Finish()
			progress.Start()
		}

		cooccurrence := make(map[uint64]float64)
		for i := 0; i < documentSize; i++ {
			for j := i + 1; j <= i+window; j++ {
				if j >= documentSize {
					continue
				}
				f, err := countValue(typ, i, j)
				if err != nil {
					return nil, errors.Wrap(err, "Failed to count co-occurrence between words")
				}
				cooccurrence[co.EncodeBigram(uint64(c.Document[i]), uint64(c.Document[j]))] += f
			}
			if verbose {
				progress.Increment()
			}
		}
		return cooccurrence, nil
	default:
		return nil, errors.Errorf("Invalid mode=%s to get co-occurrence", mode)
	}
}

// Pair stores co-occurrence information.
type Pair struct {
	// L1 and L2 store index number for two co-occurrence words.
	L1, L2 int
	// F stores the measures of co-occurrence.
	F float64
	// Coefficient stores a coefficient for weighted matrix factorization.
	Coefficient float64
}

func (c *WegoCorpus) PairsIntoGlove(mode model.Mode, window int, xmax int, alpha float64, verbose bool) ([]Pair, error) {
	cooccurrence, err := c.cooccurrence(mode, window, DISTANCE, verbose)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create Pairs for GloVe")
	}
	pairSize := len(cooccurrence)
	pairs := make([]Pair, pairSize)
	shuffle := rand.Perm(pairSize)

	var progress *pb.ProgressBar
	if verbose {
		fmt.Println("Scan cooccurrences for pairs")
		progress = pb.New(pairSize).SetWidth(80)
		defer progress.Finish()
		progress.Start()
	}

	var i int
	for p, f := range cooccurrence {
		coefficient := 1.0
		if f < float64(xmax) {
			coefficient = math.Pow(f/float64(xmax), alpha)
		}

		ul1, ul2 := co.DecodeBigram(p)
		pairs[shuffle[i]] = Pair{
			L1:          int(ul1),
			L2:          int(ul2),
			F:           math.Log(f),
			Coefficient: coefficient,
		}
		i++
		if verbose {
			progress.Increment()
		}
	}
	return pairs, nil
}

// PairMap stores co-occurrences.
type PairMap map[uint64]float64

func (c *WegoCorpus) PairsIntoLexvec(mode model.Mode, window int, relationType RelationType, smooth float64, verbose bool) (PairMap, error) {
	cooccurrence, err := c.cooccurrence(mode, window, INCREMENT, verbose)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create Pairs for Lexvec")
	}
	cooccurrenceSize := len(cooccurrence)

	var progress *pb.ProgressBar
	if verbose {
		fmt.Println("Scan cooccurrences for pairs")
		progress = pb.New(cooccurrenceSize).SetWidth(80)
		defer progress.Finish()
		progress.Start()
	}

	logTotalFreq := math.Log(math.Pow(float64(c.TotalFreq()), smooth))
	for p, f := range cooccurrence {
		ul1, ul2 := co.DecodeBigram(p)
		v, err := c.relationValue(relationType, int(ul1), int(ul2), f, logTotalFreq, smooth)
		if err != nil {
			return nil, errors.Wrap(err, "Failed to calculate relation value")
		}
		cooccurrence[p] = v
		if verbose {
			progress.Increment()
		}
	}
	return cooccurrence, nil
}

func (c *WegoCorpus) relationValue(typ RelationType, l1, l2 int, co, logTotalFreq, smooth float64) (float64, error) {
	switch typ {
	case PPMI:
		if co == 0 {
			return 0, nil
		}
		// TODO: avoid log for l1, l2 every time
		ppmi := math.Log(co) - math.Log(float64(c.IDFreq(l1))) - math.Log(math.Pow(float64(c.IDFreq(l2)), smooth)) + logTotalFreq
		if ppmi < 0 {
			ppmi = 0
		}
		return ppmi, nil
	case PMI:
		if co == 0 {
			return 1, nil
		}
		pmi := math.Log(co) - math.Log(float64(c.IDFreq(l1))) - math.Log(math.Pow(float64(c.IDFreq(l2)), smooth)) + logTotalFreq
		return pmi, nil
	case CO:
		return co, nil
	case LOGCO:
		return math.Log(co), nil
	default:
		return 0, errors.Errorf("Invalid measure type")
	}
}

func countValue(typ CountType, left, right int) (float64, error) {
	switch typ {
	case INCREMENT:
		return 1., nil
	case DISTANCE:
		div := left - right
		if div == 0 {
			return 0, errors.Errorf("Divide by zero on counting co-occurrence")
		}
		return 1. / math.Abs(float64(div)), nil
	default:
		return 0, errors.Errorf("Invalid count type")
	}
}
