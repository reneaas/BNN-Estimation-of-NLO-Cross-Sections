#!/bin/zsh

#Usage:
#>> get_lines filename keyword num_lines
function get_lines {
    filename=$1
    keyword=$2
    num_lines=$3

    start_line=$(cat ${filename} | grep ${keyword} --line-number | cut -f1 -d:) #Extract line number of found keyword
    end_line=$(( ${start_line} + ${num_lines} ))
    lines=$(sed -n -e ${start_line},${end_line}p ${filename})
    echo ${lines}
}

filename=$1
keyword=$2
num_lines=$3

start_line=$(cat ${filename} | grep ${keyword} --line-number | cut -f1 -d:) #Extract line number of found keyword
end_line=$(( ${start_line} + ${num_lines} ))
lines=$(sed -n -e ${start_line},${end_line}p ${filename})
echo ${lines}
