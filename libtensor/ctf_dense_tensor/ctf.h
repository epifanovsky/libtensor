#ifndef LIBTENSOR_CTF_H
#define LIBTENSOR_CTF_H

#include <mpi.h>
#include <cyclopstf/src/dist_tensor/cyclopstf.hpp>
#include <libutil/singleton.h>

namespace libtensor {


class ctf : public libutil::singleton<ctf> {
    friend class libutil::singleton<ctf>;

private:
    tCTF<double> m_ctf;

protected:
    ctf() { }

public:
    static tCTF<double> &get() {
        return ctf::get_instance().m_ctf;
    }

    static bool is_master() {
        int myid;
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        return myid == 0;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_H

