# impact-dashboard
This repository hosts the source code for the impact dashboard. This must be run using a Chrome browser, version compatability currently unknown. 

## Run locally


Create environment
```
conda env create -f examples/local-environment.yml
conda activate impact-dashboard-local
```

Set environment variables
```
$ export MONGO_PORT=27017
$ export MONGO_HOST=localhost
$ export OUTPUT_DIR=/path/to/impact/output
```

Start mongodb
```
$ mkdir mongodb_data # db path
$ touch mongodb_log # log data path
$ mongod --port $MONGO_PORT --dbpath mongodb_data --fork --logpath mongodb_log
```

Backfill existing results:
```
$ import-docs
```

Start monitor:
```
$ start-monitor &
```

Launch dashboard:
```
$ launch-app
```










## Docker


Running requires the setting of `MONGO_HOST`, `MONGO_PORT`, and `OUTPUT_DIR` environment variables.

```
docker run -e MONGO_PORT=$MONGO_PORT -e MONGO_HOST=$MONGO_HOST -p "8050:8050" -v $OUTPUT_DIR:/app/files" -t impact-dash
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
- [ ] Link table selection
- [ ] Change colors selection model
- [ ] make points smaller
- [ ] Clear color by selection

## Known issues
There are some quirks with the rendering.

* If the DataTables have few rows and do not fill the table height, the dashboard will fail to render with a maximum callstack javascript error due to a resizing [bug](https://github.com/plotly/dash/issues/1775).