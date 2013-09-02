set(SRC_CTF
    src/interface/ctf_world.cxx
    src/interface/ctf_tensor.cxx
    src/interface/ctf_matrix.cxx
    src/interface/ctf_scalar.cxx
    src/interface/ctf_vector.cxx
    src/interface/ctf_idx_tensor.cxx
    src/dist_tensor/cyclopstf.cxx
#    src/dist_tensor/dist_tensor.cxx
    src/shared/comm.cxx
    src/shared/memcontrol.cxx
    src/shared/timer.cxx
    src/shared/unit_util.cxx
    src/shared/util.cxx
)

include_directories(include)

add_definitions(-DINT64_MAX=9223372036854775807L)
add_definitions(-DUINT64_MAX=18446744073709551615UL)
add_library(ctf ${SRC_CTF})

