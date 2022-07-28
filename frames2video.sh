data=$1
cd $data
cwd=$(pwd)
for d in */ ; do
    echo "$d"
    cd $d
    cd color
    #for i in * ; do
    #    echo "$i"
    #done
    ffmpeg -f image2 -i frame%07d.jpg $cwd/$d.avi
    cd $cwd
    break
done
