#ifndef LIBTENSOR_SE_PART_IMPL_H
#define LIBTENSOR_SE_PART_IMPL_H

#include <algorithm>
#include "../bad_symmetry.h"
#include "../se_part.h"

namespace libtensor {


template<size_t N, typename T>
const char *se_part<N, T>::k_clazz = "se_part<N, T>";


template<size_t N, typename T>
const char *se_part<N, T>::k_sym_type = "part";


template<size_t N, typename T>
se_part<N, T>::se_part(
    const block_index_space<N> &bis,
    const mask<N> &msk,
    size_t npart) :

    m_bis(bis),
    m_bidims(m_bis.get_block_index_dims()),
    m_pdims(make_pdims(bis, msk, npart)),
    m_bipdims(make_bipdims(
        m_bis.get_block_index_dims(), make_pdims(bis, msk, npart))),
    m_fmap(m_pdims.get_size()),
    m_fmapi(m_pdims.get_size()),
    m_rmap(m_pdims.get_size()),
    m_ftr(m_pdims.get_size()) {

    size_t mapsz = m_pdims.get_size();
    for (size_t i = 0; i < mapsz; i++) {
        m_fmap[i] = m_rmap[i] = i;
        abs_index<N>::get_index(i, m_pdims, m_fmapi[i]);
    }
}


template<size_t N, typename T>
se_part<N, T>::se_part(
    const block_index_space<N> &bis,
    const dimensions<N> &pdims) :

    m_bis(bis),
    m_bidims(m_bis.get_block_index_dims()),
    m_pdims(pdims),
    m_bipdims(make_bipdims(m_bis.get_block_index_dims(), pdims)),
    m_fmap(m_pdims.get_size()),
    m_fmapi(m_pdims.get_size()),
    m_rmap(m_pdims.get_size()),
    m_ftr(m_pdims.get_size()) {

#ifdef LIBTENSOR_DEBUG
    static const char *method =
        "se_part(const block_index_space<N>&, const dimensions<N>&)";

    if(!is_valid_pdims(bis, pdims)) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__, "pdims");
    }
#endif

    size_t mapsz = m_pdims.get_size();
    for (size_t i = 0; i < mapsz; i++) {
        m_fmap[i] = m_rmap[i] = i;
        abs_index<N>::get_index(i, m_pdims, m_fmapi[i]);
    }
}


template<size_t N, typename T>
se_part<N, T>::se_part(const se_part<N, T> &elem) :

    m_bis(elem.m_bis),
    m_bidims(elem.m_bidims),
    m_pdims(elem.m_pdims),
    m_bipdims(elem.m_bipdims),
    m_fmap(elem.m_fmap),
    m_fmapi(elem.m_fmapi),
    m_rmap(elem.m_rmap),
    m_ftr(elem.m_ftr) {

}


template<size_t N, typename T>
se_part<N, T>::~se_part() {

}


template<size_t N, typename T>
void se_part<N, T>::add_map(
    const index<N> &idx1,
    const index<N> &idx2,
    const scalar_transf<T> &tr) {

    static const char *method =
        "add_map(const index<N>&, const index<N>&, scalar_transf<T>)";

#ifdef LIBTENSOR_DEBUG
    if(!is_valid_pidx(idx1)) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "idx1");
    }
    if(!is_valid_pidx(idx2)) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "idx2");
    }
#endif // LIBTENSOR_DEBUG

    size_t a = abs_index<N>::get_abs_index(idx1, m_pdims);
    size_t b = abs_index<N>::get_abs_index(idx2, m_pdims);

    if(a == b) return;
    bool swapped = false;
    if(a > b) {
        std::swap(a, b);
        swapped = true;
    }

    const index<N> &idxa = swapped ? idx2 : idx1;
    const index<N> &idxb = swapped ? idx1 : idx2;

    // If a was forbidden allow it (create a one-loop)
    if(m_fmap[a] == size_t(-1)) {
        m_fmap[a] = a;
        m_fmapi[a] = idxa;
        m_rmap[a] = a;
        m_ftr[a].reset();
    }

    // If b was forbidden allow it (create a one-loop)
    if(m_fmap[b] == size_t(-1)) {
        m_fmap[b] = b;
        m_fmapi[b] = idxb;
        m_rmap[b] = b;
        m_ftr[b].reset();
    }

    // check if b is in the same loop as a
    size_t ax = a, axf = m_fmap[ax];
    scalar_transf<T> sx;
    while(ax < axf && ax < b) {
        sx.transform(m_ftr[ax]);
        ax = axf;
        axf = m_fmap[ax];
    }
    if(ax == b) {
        if(swapped) sx.invert();
        if(sx != tr) {
            throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Mapping exists with different sign.");
        }
        return;
    }

    size_t br = m_rmap[b], bf = m_fmap[b];
    scalar_transf<T> sab(tr); // cur a -> cur b
    if(swapped) sab.invert();
    while(b != bf) {
        // remove b from its loop
        sx = m_ftr[b];
        m_fmap[br] = bf;
        abs_index<N>::get_index(bf, m_pdims, m_fmapi[br]);
        m_rmap[bf] = br;
        m_ftr[br].transform(sx);

        // add it to the loop of a
        add_to_loop(a, b, sab);

        // go to next b
        a = b; b = bf; bf = m_fmap[b];
        sab = sx;
    }

    // add last b to loop
    add_to_loop(a, b, sab);
}


template<size_t N, typename T>
void se_part<N, T>::mark_forbidden(const index<N> &idx) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "mark_forbidden(const index<N>&)";

    if(!is_valid_pidx(idx)) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "idx");
    }
#endif // LIBTENSOR_DEBUG

    size_t a = abs_index<N>::get_abs_index(idx, m_pdims);

    if(m_fmap[a] == size_t(-1)) return;

    index<N> i0;
    size_t af = m_fmap[a];
    while(af != a) {
        size_t ax = af;
        af = m_fmap[ax];
        m_fmap[ax] = size_t(-1);
        m_fmapi[ax] = i0;
        m_rmap[ax] = size_t(-1);
        m_ftr[ax].reset();
    }
    m_fmap[a] = size_t(-1);
    m_fmapi[a] = i0;
    m_rmap[a] = size_t(-1);
    m_ftr[a].reset();
}


template<size_t N, typename T>
const index<N> &se_part<N, T>::get_direct_map(const index<N> &idx) const {

    static const char *method = "get_direct_map(const index<N>&)";

    size_t aidx = abs_index<N>::get_abs_index(idx, m_pdims);

#ifdef LIBTENSOR_DEBUG
    if(m_fmap[aidx] == size_t(-1))
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Partition is not allowed.");
#endif

    return m_fmapi[aidx];
}


template<size_t N, typename T>
scalar_transf<T> se_part<N, T>::get_transf(
    const index<N> &from,
    const index<N> &to) const {

    static const char *method = "get_transf(const index<N>&, const index<N>&)";

    size_t a = abs_index<N>::get_abs_index(from, m_pdims);
    size_t b = abs_index<N>::get_abs_index(to, m_pdims);

    if(a == b) return scalar_transf<T>();

    bool swapped = false;
    if(a > b) { swapped = true; std::swap(a, b); }
    size_t x = m_fmap[a];
    scalar_transf<T> tr(m_ftr[a]);
    while(x != b && a < x) {
        tr.transform(m_ftr[x]);
        x = m_fmap[x];
    }
    if(x <= a) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
            "No mapping.");
    }

    if (swapped) tr.invert();
    return tr;
}


template<size_t N, typename T>
bool se_part<N, T>::map_exists(const index<N> &from, const index<N> &to) const {

    size_t a = abs_index<N>::get_abs_index(from, m_pdims);
    size_t b = abs_index<N>::get_abs_index(to, m_pdims);

    if(a > b) std::swap(a, b);
    if(m_fmap[a] == size_t(-1)) return false;
    if(m_fmap[b] == size_t(-1)) return false;

    size_t x = m_fmap[a];
    while(x != b && a < x) x = m_fmap[x];
    return (x == b);
}


template<size_t N, typename T>
bool se_part<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

    return m_bis.equals(bis);
}


template<size_t N, typename T>
bool se_part<N, T>::is_allowed(const index<N> &idx) const {

    index<N> pidx;
    for(register size_t i = 0; i < N; i++) {
        pidx[i] = idx[i] / m_bipdims[i];
    }

    return !is_forbidden(pidx);
}


template<size_t N, typename T>
void se_part<N, T>::permute(const permutation<N> &perm) {

    if (perm.is_identity()) return;

    m_bis.permute(perm);
    m_bidims.permute(perm);
    m_bipdims.permute(perm);

    bool affects_map = false;
    for(size_t i = 0; i < N; i++) {
        if(m_pdims[i] != 1 && perm[i] != i) { affects_map = true; break; }
    }

    if(!affects_map) return;

    dimensions<N> pdims(m_pdims);
    m_pdims.permute(perm);

    size_t mapsz = m_pdims.get_size();
    std::vector<size_t> fmap(mapsz), rmap(mapsz);
    std::vector< index<N> > fmapi(mapsz);
    std::vector< scalar_transf<T> > ftr(mapsz);

    for(size_t i = 0; i < mapsz; i++) {
        fmap[i] = rmap[i] = i;
        abs_index<N>::get_index(i, m_pdims, fmapi[i]);
    }
    std::swap(m_fmap, fmap);
    std::swap(m_fmapi, fmapi);
    std::swap(m_rmap, rmap);
    std::swap(m_ftr, ftr);

    for(size_t i = 0; i < mapsz; i++) {

        if(fmap[i] <= i) continue;

        index<N> ia;
        abs_index<N>::get_index(i, pdims, ia);
        ia.permute(perm);
        size_t naia = abs_index<N>::get_abs_index(ia, m_pdims);

        if(fmap[i] == size_t(-1)) {
            m_fmap[naia] = m_rmap[naia] = size_t(-1);
            continue;
        }

        index<N> ib;
        abs_index<N>::get_index(fmap[i], pdims, ib);
        ib.permute(perm);
        add_map(ia, ib, ftr[i]);
    }
}


template<size_t N, typename T>
void se_part<N, T>::apply(index<N> &idx) const {

    //  Determine partition index and offset within partition
    //
    index<N> pidx1;
    for(register size_t i = 0; i < N; i++) {
        pidx1[i] = idx[i] / m_bipdims[i];
    }

    //  Map the partition index
    //
    size_t apidx = abs_index<N>::get_abs_index(pidx1, m_pdims);
    if (m_fmap[apidx] == size_t(-1)) return;
    const index<N> &pidx2 = m_fmapi[apidx];

    //  Construct a mapped block index
    //
    for(register size_t i = 0; i < N; i++) {
        idx[i] -= (pidx1[i] - pidx2[i]) * m_bipdims[i];
    }
}


template<size_t N, typename T>
void se_part<N, T>::apply(index<N> &idx, tensor_transf<N, T> &tr) const {

    //  Determine partition index and offset within partition
    //
    index<N> pidx1;
    for(register size_t i = 0; i < N; i++) {
        pidx1[i] = idx[i] / m_bipdims[i];
    }

    //  Map the partition index
    //
    size_t apidx = abs_index<N>::get_abs_index(pidx1, m_pdims);
    if (m_fmap[apidx] == size_t(-1)) return;
    const index<N> &pidx2 = m_fmapi[apidx];

    //  Construct a mapped block index
    //
    for(register size_t i = 0; i < N; i++) {
        idx[i] -= (pidx1[i] - pidx2[i]) * m_bipdims[i];
    }

    tr.transform(m_ftr[apidx]);
}


template<size_t N, typename T>
dimensions<N> se_part<N, T>::make_pdims(
    const block_index_space<N> &bis,
    const mask<N> &msk,
    size_t npart) {

    static const char *method = "make_pdims(const block_index_space<N>&, "
        "const mask<N>&, size_t)";

    if(npart < 2) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__, "npart");
    }

    index<N> i1, i2;
    size_t m = 0;
    for(register size_t i = 0; i < N; i++) {
        if(msk[i]) {
            i2[i] = npart - 1;
            m++;
        } else {
            i2[i] = 0;
        }
    }

    //  Make sure the partitioning is not trivial
    //
    if(m == 0) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__, "msk");
    }

    dimensions<N> pdims(index_range<N>(i1, i2));

#ifdef LIBTENSOR_DEBUG
    if(!is_valid_pdims(bis, pdims)) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__, "bis");
    }
#endif

    return pdims;
}


template<size_t N, typename T>
dimensions<N> se_part<N, T>::make_bipdims(
    const dimensions<N> &bidims,
    const dimensions<N> &pdims) {

    static const char *method = "make_bipdims(const dimensions<N>&, "
        "const dimensions<N>&)";

    index<N> i1, i2;
    for(register size_t i = 0; i < N; i++) {
#ifdef LIBTENSOR_DEBUG
        if(bidims[i] % pdims[i] != 0) {
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                "pdims");
        }
#endif

        i2[i] = bidims[i] / pdims[i] - 1;
    }

    return dimensions<N>(index_range<N>(i1, i2));
}


template<size_t N, typename T>
bool se_part<N, T>::is_valid_pdims(
    const block_index_space<N> &bis,
    const dimensions<N> &d) {

    dimensions<N> bidims = bis.get_block_index_dims();

    //  Make sure the splits are identical for all partitions
    //
    for(size_t i = 0; i < N; i++) {

        size_t np = d[i];
        if(np == 1) continue;

        if(bidims[i] % np != 0) return false;

        size_t psz = bidims[i] / np;
        const split_points &pts = bis.get_splits(bis.get_type(i));
        size_t d = pts[psz - 1];
        for(size_t j = 0; j < psz; j++) {
            size_t pt0 = j == 0 ? 0 : pts[j - 1];
            for(size_t k = 1; k < np; k++) {
                if(pts[k * psz + j - 1] != pt0 + k * d) return false;
            }
        }
    }
    return true;
}


template<size_t N, typename T>
void se_part<N, T>::add_to_loop(size_t a, size_t b,
    const scalar_transf<T> &tr) {

    size_t af = m_fmap[a];
    scalar_transf<T> tx(tr);
    tx.invert();
    if(a < b) {
        while(af < b && a < af) {
            tx.transform(m_ftr[a]);
            a = af; af = m_fmap[a];
        }
    } else {
        while((af < b && b < a) || (a < af && af < b) || (b < a && a < af)) {
            tx.transform(m_ftr[a]);
            a = af; af = m_fmap[a];
        }
    }
    tx.transform(m_ftr[a]);
    m_fmap[a] = b; m_rmap[b] = a;
    abs_index<N>::get_index(b, m_pdims, m_fmapi[a]);
    m_fmap[b] = af; m_rmap[af] = b;
    abs_index<N>::get_index(af, m_pdims, m_fmapi[b]);
    m_ftr[b] = tx;
    m_ftr[a].transform(tx.invert());
}


template<size_t N, typename T>
bool se_part<N, T>::is_valid_pidx(const index<N> &idx) {

    for(register size_t i = 0; i < N; i++)
        if(idx[i] >= m_pdims[i]) return false;
    return true;
}


} // namespace libtensor


#endif // LIBTENSOR_SE_PART_IMPL_H
