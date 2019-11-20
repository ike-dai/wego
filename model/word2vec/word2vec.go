// Copyright © 2017 Makoto Ito
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package word2vec

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"

	"github.com/ynqa/wego/corpus"
	"github.com/ynqa/wego/model"
)

type Word2vecOption struct {
	Mod                Model
	Opt                Optimizer
	SubsampleThreshold float64
	Theta              float64
}

// Word2vec stores the configs for Word2vec models.
type Word2vec struct {
	*model.Option
	*Word2vecOption
	*corpus.Word2vecCorpus

	subSamples []float64

	// words' vector.
	vector []float64

	// manage learning rate.
	currentlr        float64
	trained          chan struct{}
	trainedWordCount int

	// manage data range per thread.
	indexPerThread []int

	// progress bar.
	progress *pb.ProgressBar
}

// NewWord2vec creates *Word2Vec.
func NewWord2vec(option *model.Option, word2vecOption *Word2vecOption) *Word2vec {
	return &Word2vec{
		Option:         option,
		Word2vecOption: word2vecOption,

		currentlr: option.Initlr,
		trained:   make(chan struct{}),
	}
}

func (w *Word2vec) initialize() error {
	// Store subsumple before training.
	w.subSamples = make([]float64, w.Word2vecCorpus.Size())
	for i := 0; i < w.Word2vecCorpus.Size(); i++ {
		z := float64(w.Word2vecCorpus.IDFreq(i)) / float64(w.Word2vecCorpus.TotalFreq())
		w.subSamples[i] = (math.Sqrt(z/w.SubsampleThreshold) + 1.0) *
			w.SubsampleThreshold / z
	}

	// Initialize word vector.
	vectorSize := w.Word2vecCorpus.Size() * w.Dimension
	w.vector = make([]float64, vectorSize)
	for i := 0; i < vectorSize; i++ {
		w.vector[i] = (rand.Float64() - 0.5) / float64(w.Dimension)
	}

	// Initialize optimizer.
	return w.Opt.initialize(w.Word2vecCorpus, w.Dimension)
}

// Train trains words' vector on corpus.
func (w *Word2vec) Train(f io.Reader) error {
	c := corpus.NewWord2vecCorpus()
	if err := c.Parse(f, w.ToLower, w.MinCount, w.BatchSize, w.Verbose); err != nil {
		return errors.Wrap(err, "Failed to parse corpus")
	}
	w.Word2vecCorpus = c
	if err := w.initialize(); err != nil {
		return errors.Wrap(err, "Failed to initialize")
	}
	return w.train()
}

func (w *Word2vec) train() error {
	document := w.Document
	documentSize := len(document)
	if documentSize <= 0 {
		return errors.New("No words for training")
	}

	w.indexPerThread = model.IndexPerThread(w.ThreadSize, documentSize)

	for i := 1; i <= w.Iteration; i++ {
		if w.Verbose {
			fmt.Printf("Train %d-th:\n", i)
			w.progress = pb.New(documentSize).SetWidth(80)
			w.progress.Start()
		}
		go w.observeLearningRate()

		semaphore := make(chan struct{}, w.ThreadSize)
		waitGroup := &sync.WaitGroup{}

		for j := 0; j < w.ThreadSize; j++ {
			waitGroup.Add(1)
			go w.trainPerThread(document[w.indexPerThread[j]:w.indexPerThread[j+1]], w.Mod.trainOne,
				semaphore, waitGroup)
		}
		waitGroup.Wait()
		if w.Verbose {
			w.progress.Finish()
		}
	}
	return nil
}

func (w *Word2vec) trainPerThread(document []int,
	trainOne func(document []int, wordIndex int, wordVector []float64, lr float64, optimizer Optimizer),
	semaphore chan struct{}, waitGroup *sync.WaitGroup) {

	defer func() {
		<-semaphore
		waitGroup.Done()
	}()

	semaphore <- struct{}{}
	for idx, wordID := range document {
		if w.Verbose {
			w.progress.Increment()
		}

		bernoulliTrial := rand.Float64()
		p := w.subSamples[wordID]
		if p < bernoulliTrial {
			continue
		}
		trainOne(document, idx, w.vector, w.currentlr, w.Opt)
		w.trained <- struct{}{}
	}
}

func (w *Word2vec) observeLearningRate() {
	for range w.trained {
		w.trainedWordCount++
		if w.trainedWordCount%w.BatchSize == 0 {
			w.currentlr = w.Initlr * (1.0 - float64(w.trainedWordCount)/float64(w.TotalFreq()))
			if w.currentlr < w.Initlr*w.Theta {
				w.currentlr = w.Initlr * w.Theta
			}
		}
	}
}

// Save saves the word vector to outputFile.
func (w *Word2vec) Save(outputPath string) error {
	extractDir := func(path string) string {
		e := strings.Split(path, "/")
		return strings.Join(e[:len(e)-1], "/")
	}

	dir := extractDir(outputPath)

	if err := os.MkdirAll("."+string(filepath.Separator)+dir, 0777); err != nil {
		return err
	}

	file, err := os.Create(outputPath)

	if err != nil {
		return err
	}
	wr := bufio.NewWriter(file)

	defer func() {
		wr.Flush()
		file.Close()
	}()

	wordSize := w.Size()
	if w.Verbose {
		fmt.Println("Save:")
		w.progress = pb.New(wordSize).SetWidth(80)
		defer w.progress.Finish()
		w.progress.Start()
	}

	var contextVector []float64
	switch opt := w.Opt.(type) {
	case *NegativeSampling:
		contextVector = opt.ContextVector
	}

	var buf bytes.Buffer
	for i := 0; i < wordSize; i++ {
		word, _ := w.Word(i)
		fmt.Fprintf(&buf, "%v ", word)
		for j := 0; j < w.Dimension; j++ {
			var v float64
			l := i*w.Dimension + j
			switch {
			case w.SaveVectorType == model.ADD && len(contextVector) != 0:
				v = w.vector[l] + contextVector[l]
			case w.SaveVectorType == model.NORMAL:
				v = w.vector[l]
			default:
				return errors.Errorf("Invalid case to save vector type=%s", w.SaveVectorType)
			}
			fmt.Fprintf(&buf, "%f ", v)
		}
		fmt.Fprintln(&buf)
		if w.Verbose {
			w.progress.Increment()
		}
	}
	wr.WriteString(fmt.Sprintf("%v", buf.String()))
	return nil
}

// Get gets the word vector.
func (w *Word2vec) Get() (map[string][]float64, error) {
	wordSize := w.Size()
	var contextVector []float64
	switch opt := w.Opt.(type) {
	case *NegativeSampling:
		contextVector = opt.ContextVector
	}
	wordVector := make(map[string][]float64)

	for i := 0; i < wordSize; i++ {
		word, _ := w.Word(i)
		for j := 0; j < w.Dimension; j++ {
			var v float64
			l := i*w.Dimension + j
			switch {
			case w.SaveVectorType == model.ADD && len(contextVector) != 0:
				v = w.vector[l] + contextVector[l]
			case w.SaveVectorType == model.NORMAL:
				v = w.vector[l]
			default:
				return nil, errors.Errorf("Invalid case to get vector type=%s", w.SaveVectorType)
			}
			wordVector[word] = append(wordVector[word], v)
		}
	}
	return wordVector, nil
}
