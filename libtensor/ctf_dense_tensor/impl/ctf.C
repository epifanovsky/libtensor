#include "../ctf.h"
#include "ctf_world.h"

namespace libtensor {


void ctf::init() {
    ctf_world::init();
}


void ctf::exit() {
    ctf_world::exit();
}


unsigned ctf::get_rank() {
    return ctf_world::get_rank();
}


unsigned ctf::get_size() {
    return ctf_world::get_size();
}


bool ctf::is_master() {
    return ctf_world::is_master();
}


unsigned ctf::ctf_version() {
    return CTF_VERSION;
}


} // namespace libtensor

