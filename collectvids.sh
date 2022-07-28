data=$1
filename=$2
cd $data
cwd=$(pwd)
echo $cwd
touch $2
for d in */ ; do
    echo "$d"
    #$d"color_video.avi" > $cwd/$2
    printf "file '%s%s'\n" $d "color_vid.avi" >> $cwd/$2 
    cd $cwd
    #break
done
