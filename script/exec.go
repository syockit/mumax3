package script

import (
	"bytes"
	"fmt"
	"io"
	"log"
)

// First compiles the entire source, then executes it.
// So compile errors are spotted before executing the previous code.
func (p *Parser) Exec(src io.Reader) (err error) {
	defer func() {
		panc := recover()
		if panc != nil {
			err = fmt.Errorf("%v", panc)
		}
	}()

	var code []Expr
	code, err = p.parse(src)
	if err != nil {
		return
	}

	for _, e := range code {
		ret := e.Eval()
		log.Println("eval", e, ":", ret)
	}
	return nil
}

// Like Exec but takes string input
func (p *Parser) ExecString(code string) error {
	return p.Exec(bytes.NewBufferString(code))
}

type nop struct{}

func (e *nop) Eval() interface{} {
	return nil
}

func (e *nop) String() string {
	return ";"
}

type List []Expr

func (l List) Eval() interface{} {
	ret := make([]interface{}, len(l))
	for i := range ret {
		ret[i] = l[i].Eval()
	}
	return ret
}

type call struct {
	funcname string
	args     []Expr
}

func (e *call) Eval() interface{} {
	return nil
}

func (e *call) String() string {
	str := fmt.Sprint(e.funcname, "( ")
	for _, a := range e.args {
		str += fmt.Sprint(a, " ")
	}
	str += ")"
	return str
}

type num float64

func (n num) Eval() interface{} {
	return float64(n)
}

func (n num) String() string {
	return fmt.Sprint(n.Eval())
}