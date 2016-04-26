#ifndef LIBTENSOR_CTF_H
#define LIBTENSOR_CTF_H

namespace libtensor {


class ctf {
public:
    static void init();
    static void exit();
    static unsigned get_rank();
    static unsigned get_size();
    static bool is_master();
    static unsigned ctf_version();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_H

