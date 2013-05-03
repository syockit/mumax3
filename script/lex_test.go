package script

import (
	"bytes"
	"fmt"
	"testing"
)

func TestLexer(t *testing.T) {
	src := bytes.NewBuffer([]byte(testText))
	l := newLexer(src)

	for item := l.next(); item.typ != EOF; item = l.next() {
		fmt.Println(item)
	}
}

const testText = `
	alpha=1
	m.set("/home/arne/m.dump")
	run(1e-9+i)
	b=sin(2*pi*f*t)
`
