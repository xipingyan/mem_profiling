# how to usage heaptrack

```
cd heaptrack/testprj
mkdir build && cd build
cmake .. && make
$ heaptrack testprj/build/testapp

cd ../../../
$ numactl -C 0 heaptrack testprj/build/testapp
```
