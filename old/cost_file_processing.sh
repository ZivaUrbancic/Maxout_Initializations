#!/bin/bash

sed -i 's/ //g' $1              # remove whitespace (decreses file size)
sed -i 's/\])\]/\])\],/g' $1    # add commas after vector of each run
sed -i '$ s/.$//' $1            # remove comma at the end of the file

