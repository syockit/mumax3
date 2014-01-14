package oommf

import (
	"bufio"
	"fmt"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"io"
	"os"
	"strings"
)

// Read any OOMMF file, autodetect OVF1/OVF2 format
func Read(fname string) (s *data.Slice, meta data.Meta, err error) {
	in_, err := os.Open(fname)
	util.FatalErr(err)
	in := fullReader{bufio.NewReader(in_)}

	info := readHeader(in)

	n := info.Size
	c := info.StepSize
	if c == [3]float32{0, 0, 0} {
		c = [3]float32{1, 1, 1} // default (presumably unitless) cell size
	}
	data_ := data.NewSlice(3, n)

	switch info.OOMMF {
	default:
		panic("Unknown format: " + info.OOMMF)
	}
	return data_, data.Meta{Time: info.TotalTime, Unit: info.ValueUnit}, nil
}

// omf.Info represents the header part of an omf file.
// TODO: add Err to return error status
// Perhaps CheckErr() func
type Info struct {
	Desc            map[string]interface{}
	Size            [3]int
	ValueMultiplier float32
	ValueUnit       string
	Format          string // binary or text
	OVFVersion      int
	TotalTime       float64
	StageTime       float64
	DataFormat      string // 4 or 8
	StepSize        [3]float32
	MeshUnit        string
	OOMMF           string
}

// Parses the header part of the OVF1/OVF2 file
func readHeader(in io.Reader) *Info {
	desc := make(map[string]interface{})
	info := new(Info)
	info.Desc = desc

	line, eof := readLine(in)
	switch line {
	default:
		panic("unknown header: " + line)

	}
	line, eof = readLine(in)
	for !eof && !isHeaderEnd(line) {
		key, value := parseHeaderLine(line)

		switch strings.ToLower(key) {
		default:
			panic("Unknown key: " + key)
			// ignored
		case "oommf", "segment count", "begin", "title", "meshtype", "xbase", "ybase", "zbase", "xstepsize", "ystepsize", "zstepsize", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "valuerangeminmag", "valuerangemaxmag", "end":
		case "xnodes":
			info.Size[X] = atoi(value)
		case "ynodes":
			info.Size[Y] = atoi(value)
		case "znodes":
			info.Size[Z] = atoi(value)
		case "valuemultiplier":
		case "valueunit":
		case "meshunit":
			// desc tags: parse further and add to metadata table
		case "desc":
			strs := strings.SplitN(value, ":", 2)
			desc_key := strings.Trim(strs[0], "# ")
			// Desc tag does not neccesarily have a key:value layout.
			// If not, we use an empty value string.
			desc_value := ""
			if len(strs) > 1 {
				desc_value = strings.Trim(strs[1], "# ")
			}
			// 			fmt.Println(desc_key, " : ", desc_value)
			desc[desc_key] = desc_value
		}

		line, eof = readLine(in)
	}
	// the remaining line should now be the begin:data clause
	key, value := parseHeaderLine(line)
	value = strings.TrimSpace(value)
	strs := strings.SplitN(value, " ", 3)
	if strings.ToLower(key) != "begin" || strings.ToLower(strs[0]) != "data" {
		panic("Expected: Begin: Data")
	}
	info.Format = strings.ToLower(strs[1])
	if len(strs) >= 3 { // dataformat for text is empty
		info.DataFormat = strs[2]
	}
	return info
}

// INTERNAL: Splits "# key: value" into "key", "value".
// Both may be empty
func parseHeaderLine(str string) (key, value string) {
	strs := strings.SplitN(str, ":", 2)
	key = strings.Trim(strs[0], "# ")
	if len(strs) != 2 {
		return key, ""
	}
	value = strings.Trim(strs[1], "# ")
	return key, value
}

// INTERNAL: true if line == "# begin_data"
func isHeaderEnd(str string) bool {
	str = strings.ToLower(strings.Trim(str, "# "))
	str = strings.Replace(str, " ", "", -1)
	return strings.HasPrefix(str, "begin:data")
}

const OVF_CONTROL_NUMBER_4 = 1234567.0 // The omf format requires the first encoded number in the binary data section to be this control number

// read data block in text format, for OVF1 and OVF2
func readOVFText(in io.Reader, t *data.Slice) {
	size := t.Size()
	data := t.Tensors()
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < 3; c++ {
					_, err := fmt.Fscan(in, &data[c][iz][iy][ix])
					if err != nil {
						panic(err)
					}
				}
			}
		}
	}
}

// write data block in text format, for OVF1 and OVF2
func writeOVFText(out io.Writer, tens *data.Slice) (err error) {
	data := tens.Tensors()
	gridsize := tens.Size()
	ncomp := tens.NComp()

	// Here we loop over X,Y,Z, not Z,Y,X, because
	// internal in C-order == external in Fortran-order
	for iz := 0; iz < gridsize[Z]; iz++ {
		for iy := 0; iy < gridsize[Y]; iy++ {
			for ix := 0; ix < gridsize[Z]; ix++ {
				for c := 0; c < ncomp; c++ {
					_, err = fmt.Fprint(out, data[c][iz][iy][ix], " ")
				}
				_, err = fmt.Fprint(out, "\n")
			}
		}
	}
	return
}

// Writes a header key/value pair to out:
// # Key: Value
func hdr(out io.Writer, key string, value ...interface{}) {
	_, err := fmt.Fprint(out, "# ", key, ": ")
	util.FatalErr(err, "while reading OOMMF header")
	_, err = fmt.Fprintln(out, value)
	util.FatalErr(err, "while reading OOMMF header")
}

func dsc(out io.Writer, k, v interface{}) {
	hdr(out, "Desc", k, ": ", v)
}
