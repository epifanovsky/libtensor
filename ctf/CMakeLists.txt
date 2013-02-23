set(SRC_CTF
    src/dist_tensor/cyclopstf.cxx
    src/dist_tensor/dist_tensor.cxx
    src/shared/comm.cxx
    src/shared/memcontrol.cxx
    src/shared/unit_util.cxx
    src/shared/util.cxx
)

add_definitions(-DINT64_MAX=9223372036854775807L)
add_definitions(-DUINT64_MAX=18446744073709551615UL)
add_library(ctf ${SRC_CTF})

