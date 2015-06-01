#ifndef LIBTENSOR_CTF_H
#define LIBTENSOR_CTF_H

#include <ctf.hpp>
#include <libutil/singleton.h>

namespace libtensor {


class ctf : public libutil::singleton<ctf> {
    friend class libutil::singleton<ctf>;

private:
    CTF::World *m_world;

protected:
    ctf() { }

public:
    static void init() {
        ctf::get_instance().do_init();
    }

    static void exit() {
        ctf::get_instance().do_exit();
    }

    static CTF::World &get_world() {
        return *ctf::get_instance().m_world;
    }

    static unsigned get_rank() {
        int myid;
        MPI_Comm_rank(ctf::get_instance().m_world->comm, &myid);
        return myid;
    }

    static unsigned get_size() {
        int size;
        MPI_Comm_size(ctf::get_instance().m_world->comm, &size);
        return size;
    }

    static bool is_master() {
        return get_rank() == 0;
    }

private:
    void do_init() {
        m_world = new CTF::World;
    }

    void do_exit() {
        delete m_world;
        m_world = 0;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_H

