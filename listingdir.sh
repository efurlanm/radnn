#!/bin/bash
tree -I "*§*" -I "assets" -h src -H src --nolinks -T 'src/ directory tree' > 'src/listingsrc.html'
cd
tree -I "*§*" -h data -H data --nolinks -T 'data/ directory tree' > '/home/x/git/radnn/src/listingdata.html'
