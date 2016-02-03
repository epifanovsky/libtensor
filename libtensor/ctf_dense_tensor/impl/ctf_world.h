#ifndef LIBTENSOR_CTF_WORLD_H
#define LIBTENSOR_CTF_WORLD_H

#include <ctf.hpp>
#include <libutil/singleton.h>

namespace libtensor {


class ctf_world : public libutil::singleton<ctf_world> {
    friend class libutil::singleton<ctf_world>;

private:
    CTF::World *m_world;
    int m_rank;
    int m_size;

protected:
    ctf_world() { }

public:
    static void init() {
        ctf_world::get_instance().do_init();
    }

    static void exit() {
        ctf_world::get_instance().do_exit();
    }

    static CTF::World &get_world() {
        return *ctf_world::get_instance().m_world;
    }

    static unsigned get_rank() {
        return ctf_world::get_instance().m_rank;
    }

    static unsigned get_size() {
        return ctf_world::get_instance().m_size;
    }

    static bool is_master() {
        return get_rank() == 0;
    }

private:
    void do_init() {
        m_world = new CTF::World;
        MPI_Comm_size(ctf_world::get_instance().m_world->comm, &m_size);
        MPI_Comm_rank(ctf_world::get_instance().m_world->comm, &m_rank);
    }

    void do_exit() {
        delete m_world;
        m_world = 0;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_WORLD_H

