for dirpath in files/*/; do
    set -- "$dirpath"/*.avi
    ffmpeg -i "concat:$1|$2" -c copy "$dirpath/both_${1##*/}"
done
