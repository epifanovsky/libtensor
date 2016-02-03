#ifndef LIBTENSOR_CTF_SYMMETRY_BUILDER_H
#define LIBTENSOR_CTF_SYMMETRY_BUILDER_H

#include <libtensor/core/transf_list.h>
#include <libtensor/ctf_dense_tensor/ctf_symmetry.h>

namespace libtensor {


template<size_t N, typename T>
class ctf_symmetry_builder {
private:
    ctf_symmetry<N, T> m_sym;

public:
    ctf_symmetry_builder(const transf_list<N, T> &trl) :
        m_sym(build(trl))
    { }

    const ctf_symmetry<N, T> &get_symmetry() const {
        return m_sym;
    }

private:
    static ctf_symmetry<N, T> build(const transf_list<N, T> &trl);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_SYMMETRY_BUILDER_H
