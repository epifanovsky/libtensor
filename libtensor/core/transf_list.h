#ifndef LIBTENSOR_TRANSF_LIST_H
#define LIBTENSOR_TRANSF_LIST_H

#include <algorithm>
#include <list>
#include <map>
#include <vector>
#include "../timings.h"
#include "abs_index.h"
#include "symmetry.h"
#include "tensor_transf.h"

namespace libtensor {


/** \brief Enumerates all transformations associated with a block in
        a symmetry group
    \tparam N Tensor order (symmetry cardinality).
    \tparam T Tensor element type.

    This algorithm applies all elements in a given symmetry group
    until it exhausts all the transformations associated with a given
    block.

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class transf_list : public timings< transf_list<N, T> > {
public:
    static const char k_clazz[]; //!< Class name

private:
    typedef std::list< tensor_transf<N, T> > transf_lst_t;
    typedef std::map< size_t, transf_lst_t > transf_map_t;

public:
    typedef typename transf_lst_t::const_iterator
        iterator; //!< List iterator

private:
    transf_lst_t m_trlist;

public:
    /** \brief Constructs the list of transformations
        \param sym Symmetry group.
        \param idx Block %index.
     **/
    transf_list(const symmetry<N, T> &sym, const index<N> &idx);

    /** \brief Returns true if the transformation is listed,
            false otherwise
     **/
    bool is_found(const tensor_transf<N, T> &tr) const;

    //!    \name STL-like list iterator
    //@{

    iterator begin() const {

        return m_trlist.begin();
    }

    iterator end() const {

        return m_trlist.end();
    }

    const tensor_transf<N, T> &get_transf(iterator &i) const {

        return *i;
    }

    //@}

private:
    bool is_found(const transf_lst_t &trlist,
            const tensor_transf<N, T> &tr) const;

    void visit(const symmetry<N, T> &sym, const abs_index<N> &aidx,
        const tensor_transf<N, T> &tr, transf_map_t &visited);

};


template<size_t N, typename T>
const char transf_list<N, T>::k_clazz[] = "transf_list<N, T>";


template<size_t N, typename T>
transf_list<N, T>::transf_list(const symmetry<N, T> &sym, const index<N> &idx) {

    transf_list<N, T>::start_timer();

    abs_index<N> aidx(idx, sym.get_bis().get_block_index_dims());
    transf_map_t visited;
    visit(sym, aidx, tensor_transf<N, T>(), visited);
    m_trlist.splice(m_trlist.end(), visited[aidx.get_abs_index()]);

    transf_list<N, T>::stop_timer();
}


template<size_t N, typename T>
bool transf_list<N, T>::is_found(const tensor_transf<N, T> &tr) const {

    return is_found(m_trlist, tr);
}


template<size_t N, typename T>
bool transf_list<N, T>::is_found(const transf_lst_t &trlist,
    const tensor_transf<N, T> &tr) const {

    return std::find(trlist.begin(), trlist.end(), tr) != trlist.end();
}


template<size_t N, typename T>
void transf_list<N, T>::visit(const symmetry<N, T> &sym,
    const abs_index<N> &aidx, const tensor_transf<N, T> &tr,
    transf_map_t &visited) {

    transf_lst_t &lst = visited[aidx.get_abs_index()];
    if(is_found(lst, tr)) return;
    lst.push_back(tr);

    for(typename symmetry<N, T>::iterator iset = sym.begin();
        iset != sym.end(); ++iset) {

        const symmetry_element_set<N, T> &eset = sym.get_subset(iset);
        for(typename symmetry_element_set<N, T>::const_iterator ielem =
            eset.begin(); ielem != eset.end(); ++ielem) {

            const symmetry_element_i<N, T> &elem = eset.get_elem(ielem);

            index<N> idx2(aidx.get_index());
            tensor_transf<N, T> tr2(tr);
            elem.apply(idx2, tr2);
            abs_index<N> aidx2(idx2, aidx.get_dims());

            visit(sym, aidx2, tr2, visited);
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_TRANSF_LIST_H
