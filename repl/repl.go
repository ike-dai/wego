// Copyright © 2019 Makoto Ito
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

package repl

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"

	"github.com/peterh/liner"
	"github.com/pkg/errors"

	"github.com/ike-dai/wego/search"
)

type Repl struct {
	searcher *search.Searcher
	line     *liner.State
	vector   []float64
	k        int
}

func NewRepl(f io.Reader, k int) (*Repl, error) {
	searcher, err := search.NewSearcher(f)
	if err != nil {
		return nil, err
	}
	line := liner.NewLiner()
	vector := make([]float64, searcher.Dimension)
	return &Repl{
		searcher: searcher,
		line:     line,
		vector:   vector,
		k:        k,
	}, nil
}

func (r *Repl) Run() error {
	defer r.line.Close()
	for {
		l, err := r.line.Prompt(">> ")
		if err != nil {
			fmt.Println("error: ", err)
		}
		switch l {
		case "exit":
			return nil
		case "":
			continue
		default:
			if err := r.eval(l); err != nil {
				fmt.Println(err)
			}
		}
	}
}

func (r *Repl) eval(l string) error {
	defer func() {
		r.vector = make([]float64, r.searcher.Dimension)
	}()

	expr, err := parser.ParseExpr(l)
	if err != nil {
		return err
	}

	var neighbors search.Neighbors
	switch e := expr.(type) {
	case *ast.Ident:
		neighbors, err = r.searcher.SearchWithQuery(e.String(), r.k)
		if err != nil {
			fmt.Printf("failed to search with word=%s\n", e.String())
		}
		return search.Describe(neighbors)
	case *ast.BinaryExpr:
		if err := r.evalExpr(expr); err != nil {
			return err
		}
		neighbors, err := r.searcher.Search(r.vector, r.k)
		if err != nil {
			fmt.Printf("failed to search with vector=%v\n", r.vector)
		}
		return search.Describe(neighbors)
	default:
		return errors.Errorf("invalid type %v", e)
	}

}

func (r *Repl) evalExpr(expr ast.Expr) error {
	switch e := expr.(type) {
	case *ast.BinaryExpr:
		return r.evalBinaryExpr(e)
	case *ast.Ident:
		return nil
	default:
		return errors.Errorf("invalid type %v", e)
	}
}

func (r *Repl) evalBinaryExpr(expr *ast.BinaryExpr) error {
	if err := r.evalExpr(expr.X); err != nil {
		return err
	}

	if err := r.evalExpr(expr.Y); err != nil {
		return err
	}

	x, ok := expr.X.(*ast.Ident)
	if ok && isZeros(r.vector) {
		xv, ok := r.searcher.Vectors[x.String()]
		if !ok {
			return errors.Errorf("not found word=%s in vector map", x.String())
		}
		copy(r.vector, xv)
	}

	y, ok := expr.Y.(*ast.Ident)
	if !ok {
		return errors.Errorf("failed to parse %v", expr.Y)
	}

	yv, ok := r.searcher.Vectors[y.String()]
	if !ok {
		return errors.Errorf("not found word=%s in vector map", y.String())
	}

	var err error
	r.vector, err = arithmetic(r.vector, expr.Op, yv)
	return err
}

func arithmetic(v1 []float64, op token.Token, v2 []float64) ([]float64, error) {
	switch op {
	case token.ADD:
		v1 = Add(v1, v2)
	case token.SUB:
		v1 = Sub(v1, v2)
	default:
		return nil, errors.Errorf("invalid operator %v", op.String())
	}
	return v1, nil
}

func isZeros(vec []float64) bool {
	for _, v := range vec {
		if v != 0. {
			return false
		}
	}
	return true
}
