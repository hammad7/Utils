# Eg: python script.py, to edit/test this it might load a "model" that takes lot of time/ram
# hence it can't be reloaded again and again
# so move the "model" to a script heavy.py and create the remaining code of script.py as remaining.py

# where "model" is used in remaining.py add this: from heavy import model
# Now type python for interactive
import imp  #p3- importlib
from heavy import *
import remaining
from remaining import *
app.run(host= '0.0.0.0', port=5050)

## Now when you edit some code in remaining.py just do this:
imp.reload(remaining) #load into current namespace
# the line from heavy import model reads model from cache
from remaining import *
app.run(host= '0.0.0.0', port=5050)

##  No more wasting time, simple debugging!!
########
