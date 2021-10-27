# impact-dashboard
This repository hosts the source code for the impact dashboard. This must be run using a Chrome browser, version compatability currently unknown. The dashboard has been tested against Chrome=v94. The rendering of LaTeX labels is broken in Firefox at present.

Running requires the setting of `MONGO_HOST` and `MONGO_PORT` environment variables.

```
docker run -e MONGO_PORT=27017 -e MONGO_HOST=172.20.3.134 -p "8050:8050" -v "/Users/jgarra/sandbox/impact/output-files:/app/files" -t impact-dash
```

## TODO:
- [x] Add optional color by column selection. Use linear color mapping, viridis, jet etc.
 add optional coloring toggle to plotly menu?  
- [x] Drop color map on selection  
- [x] Add input/output tables RHS  
- [x] Default plots
- [ ] Allow selection of all vars x, y- color input/oututs to differentiate. Note: not so simple, dbc dropdownmenu uses items with individual ids
- [x] Add remove plot option
- [x] Drop wsgi app and just use Flask
- [x] Fix color selection bug
- [x] Labels for dropdowns
- [x] Labels-> aliasing
- [x] master configuration
- [x] Dark mode style 
- [x] Remove breaks after addition
- [x] Callback for updating tables
- [x] Timed callback for updating DF
- [x] Add exploration table
- [x] sig digits explore table
- [x] sig digits value table
- [x] Check out distgen:xu_dist:file to see if material changes (does not)
- [x] exclude stop
- [x] Add archive file back to side tables
- [x] Group table items intelligently, like all magnets together
- [x] Move dynamic-input/ dynamic-output to x and y
- [x] Fix header overlap datatable
- [x] Use Chris's defaults for the dashboard
- [x] Fix regular text in latex
- [x] Fix callstack error
- [ ] Add label aliases to explore table columns
- [ ] Cache flushing
- [ ] norm_emit_z, norm_emit_xy not rendering
- [x] Update openPMD installation
- [ ] Add port to args for launch
- [ ] Dockerize
- [ ] Passable configuration file for app rendering

## Known issues
There are some quirks with the rendering.

* If the DataTables have few rows and do not fill the table height, the dashboard will fail to render with a maximum callstack javascript error due to a resizing [bug](https://github.com/plotly/dash/issues/1775).