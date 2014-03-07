#ifndef LIBTENSOR_CTF_H
#define LIBTENSOR_CTF_H

#include <ctf.hpp>
#include <libutil/singleton.h>

namespace libtensor {


class ctf : public libutil::singleton<ctf> {
    friend class libutil::singleton<ctf>;

private:
    tCTF_World<double> *m_world;

protected:
    ctf() { }

public:
    static void init() {
        ctf::get_instance().do_init();
    }

    static void exit() {
        ctf::get_instance().do_exit();
    }

    static tCTF_World<double> &get_world() {
        return *ctf::get_instance().m_world;
    }

    static bool is_master() {
        int myid;
        MPI_Comm_rank(ctf::get_instance().m_world->comm, &myid);
        return myid == 0;
    }

private:
    void do_init() {
        m_world = new tCTF_World<double>();
    }

    void do_exit() {
        delete m_world;
        m_world = 0;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_H

