#!/bin/bash
NOTEBOOKPORT=1234
ssh -N -f -R $NOTEBOOKPORT:localhost:$NOTEBOOKPORT Ricardo.Montanana@galgo.uclm.es
jupyter lab --port=$NOTEBOOKPORT --no-browser