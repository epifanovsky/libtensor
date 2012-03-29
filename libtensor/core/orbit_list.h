#ifndef LIBTENSOR_ORBIT_LIST_H
#define LIBTENSOR_ORBIT_LIST_H

#include <map>
#include <vector>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "dimensions.h"
#include "index.h"
#include "symmetry.h"

namespace libtensor {


template<size_t N, typename T>
class orbit_list : public timings< orbit_list<N, T> > {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename std::map< size_t, index<N> >::const_iterator iterator;

private:
    dimensions<N> m_dims;
    std::map< size_t, index<N> > m_orb;

public:
    orbit_list(const symmetry<N, T> &sym);
    size_t get_size() const;
    bool contains(const index<N> &idx) const;
    bool contains(size_t absidx) const;
    iterator begin() const;
    iterator end() const;
    size_t get_abs_index(iterator &i) const;
    const index<N> &get_index(iterator &i) const;

private:
    bool mark_orbit(const symmetry<N, T> &sym, const index<N> &idx,
        std::vector<char> &chk);
};


} // namespace libtensor


#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class orbit_list<1, double>;
    extern template class orbit_list<2, double>;
    extern template class orbit_list<3, double>;
    extern template class orbit_list<4, double>;
    extern template class orbit_list<5, double>;
    extern template class orbit_list<6, double>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "orbit_list_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_ORBIT_LIST_H
