package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"html/template"
	"net/http"
	"sync"
)

var ui *guistate = &guistate{Steps: 1000, Runtime: 1e-9, Cond: sync.NewCond(new(sync.Mutex))}

func gui(w http.ResponseWriter, r *http.Request) {
	if ui.templ == nil {
		ui.Lock()
		ui.templ = template.Must(template.New("gui").Parse(templText))
		ui.Heun = Solver
		ui.Mesh = Mesh()
		ui.Unlock()
	}
	util.FatalErr(ui.templ.Execute(w, ui))
}

type guistate struct {
	*cuda.Heun
	*data.Mesh
	Msg                 string
	Steps               int
	Runtime             float64
	templ               *template.Template
	Running, pleaseStop bool
	*sync.Cond
}

func (s *guistate) Time() float32 { return float32(Time) }
func (s *guistate) Lock()         { ui.L.Lock() }
func (s *guistate) Unlock()       { ui.L.Unlock() }

const templText = `
<!DOCTYPE html>
<html>
<head>
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
	{{if .Running}}<meta http-equiv="refresh" content="1">{{end}}
	<title>mx3</title>
	<style media="screen" type="text/css">
		body { margin: 40px; font-family: Helvetica, Arial, sans-serif; font-size: 16px; }
		img  { margin: 10px; }
		h1   { font-size: 28px; font-color: gray; }
		hr   { border-style: none; border-top: 1px solid gray; }
		a    { color: #375EAB; text-decoration: none; }
		table{ border:"10"; }
		div#header{ color:gray; font-size:16px; }
		div#footer{ color:gray; font-size:14px; }
	</style>
</head>

<body>

<div id="header"> <h1> mx3 </h1> <hr/> </div>

<div> <h2> control </h2>
	{{if .Running}}Running{{else}}Paused{{end}}
	{{with .Msg}}{{.}}{{end}}<br/>
	<form action=/ctl/pause method="POST"> <input type="submit" value="Pause"/> </form>
	<form action=/ctl/run   method="POST">
        <input name="value" value="{{.Runtime}}"> s <input type="submit" value="Run"/>
	</form>
	<form action=/ctl/steps method="POST">
        <input name="value" value="{{.Steps}}"> <input type="submit" value="Steps"/>
	</form>
	<b>Danger Zone:</b><form action=/ctl/exit  method="POST"> <input type="submit" value="Kill"/> </form>
<hr/></div>

<h2> solver </h2> 
<table> 
<tr> <td>     step:</td> <td>{{.NSteps}}  </td><td>  undone steps:</td> <td>{{.NUndone}}</td> </tr>
<tr> <td>     time:</td> <td>{{.Time}} s   </td><td>     time step:</td> <td>{{.Dt_si}} s </td> </tr>
<tr> <td> err/step:</td> <td>{{.LastErr}} </td><td>  max err/step:</td> <td>{{.MaxErr}}</td> </tr>
</table>
<hr/>

<h2> magnetization </h2> 
<img src="/render/m">
<hr/>


<h2> mesh </h2> 
<table> 
<tr> <td> grid size: </td> <td>{{index .Size 2}}     x{{index .Size 1}}     x{{index .Size 0}}      </td></tr>
<tr> <td> cell size: </td> <td>{{index .CellSize 2}} m x{{index .CellSize 1}} m x {{index .CellSize 0}} m </td></tr>
<tr> <td> world size:</td> <td>{{index .WorldSize 2}} m x{{index .WorldSize 1}} m x{{index .WorldSize 0}} m</td></tr>
</table>
<hr/>


</div>
<div id="footer">
	
</div>

</body>
</html>
`