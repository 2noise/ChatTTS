#!/bin/sh

exitcode=0

for file in tests/*.py
do
    python "$file"
    if [ $? -ne 0 ]
    then
        echo "Error: $file exited with a non-zero status."
        exitcode=1
    fi
done

exit $exitcode
