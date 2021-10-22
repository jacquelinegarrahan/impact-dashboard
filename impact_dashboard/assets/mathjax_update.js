function redraw(){
    var figs = document.getElementsByClassName("js-plotly-plot dash-graph--pending")
    for (var i = 0; i<figs.length; i++) {
      Plotly.redraw(figs[i])
    }
  }

setTimeout(function(){
  redraw();
}, 100);

setInterval(function() {
  MathJax.Hub.Queue(['Typeset',MathJax.Hub])
}, 1000);