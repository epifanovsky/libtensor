${BUILD_PREFIX}/bin/cmake \
        -Bbuild \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=${CXX} \
        -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
        -DBUILD_SHARED_LIBS=ON \
        -DINSTALL_DEVEL_HEADERS=ON \
        -DCMAKE_PREFIX_PATH="${PREFIX}" \
        -DBLA_VENDOR="Intel10_64lp_seq" \
        -GNinja

cd build
ninja
ctest
ninja install
