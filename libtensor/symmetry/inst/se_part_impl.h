#ifndef LIBTENSOR_SE_PART_IMPL_H
#define LIBTENSOR_SE_PART_IMPL_H

#include <algorithm>
#include <libtensor/defs.h>
#include "../bad_symmetry.h"


namespace libtensor {


template<size_t N, typename T>
const char *se_part<N, T>::k_clazz = "se_part<N, T>";


template<size_t N, typename T>
const char *se_part<N, T>::k_sym_type = "part";


template<size_t N, typename T>
se_part<N, T>::se_part(const block_index_space<N> &bis,
        const mask<N> &msk, size_t npart) :
        m_bis(bis), m_bidims(m_bis.get_block_index_dims()),
        m_pdims(make_pdims(bis, msk, npart)), m_fmap(0), m_rmap(0), m_ftr(0) {

    static const char *method =
        "se_part(const block_index_space<N>&, const mask<N>&, size_t)";

    size_t mapsz = m_pdims.get_size();
    m_fmap = new size_t[mapsz];
    m_rmap = new size_t[mapsz];
    m_ftr = new scalar_transf<T>[mapsz];
    for (size_t i = 0; i < mapsz; i++) {
        m_fmap[i] = m_rmap[i] = i;
    }
}


template<size_t N, typename T>
se_part<N, T>::se_part(const block_index_space<N> &bis,
        const dimensions<N> &pdims) :
        m_bis(bis), m_bidims(m_bis.get_block_index_dims()), m_pdims(pdims),
        m_fmap(0), m_rmap(0), m_ftr(0) {

    static const char *method =
        "se_part(const block_index_space<N>&, const dimensions<N>&)";

#ifdef LIBTENSOR_DEBUG
    if (! is_valid_pdims(bis, pdims)) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__, "pdims");
    }
#endif

    size_t mapsz = m_pdims.get_size();
    m_fmap = new size_t[mapsz];
    m_rmap = new size_t[mapsz];
    m_ftr = new scalar_transf<T>[mapsz];
    for (size_t i = 0; i < mapsz; i++) {
        m_fmap[i] = m_rmap[i] = i;
    }
}


template<size_t N, typename T>
se_part<N, T>::se_part(const se_part<N, T> &elem) :
    m_bis(elem.m_bis), m_bidims(elem.m_bidims), m_pdims(elem.m_pdims),
    m_fmap(0), m_ftr(0) {

    size_t mapsz = m_pdims.get_size();
    m_fmap = new size_t[mapsz];
    m_rmap = new size_t[mapsz];
    m_ftr = new scalar_transf<T>[mapsz];
    for (size_t i = 0; i < mapsz; i++) {
        m_fmap[i] = elem.m_fmap[i];
        m_rmap[i] = elem.m_rmap[i];
        m_ftr[i] = elem.m_ftr[i];
    }
}


template<size_t N, typename T>
se_part<N, T>::~se_part() {

    delete [] m_fmap; m_fmap = 0;
    delete [] m_rmap; m_rmap = 0;
    delete [] m_ftr; m_ftr = 0;
}


template<size_t N, typename T>
void se_part<N, T>::add_map(const index<N> &idx1,
        const index<N> &idx2, const scalar_transf<T> &tr) {

    static const char *method =
        "add_map(const index<N>&, const index<N>&, scalar_transf<T>)";

#ifdef LIBTENSOR_DEBUG
    if(!is_valid_pidx(idx1)) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                            "idx1");
    }
    if(!is_valid_pidx(idx2)) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                            "idx2");
    }
#endif // LIBTENSOR_DEBUG

    abs_index<N> aidx1(idx1, m_pdims), aidx2(idx2, m_pdims);
    size_t a = aidx1.get_abs_index(), b = aidx2.get_abs_index();

    if(a == b) return;
    bool swapped = false;
    if(a > b) {
        std::swap(a, b);
        swapped = true;
    }

    // If a was forbidden allow it (create a one-loop)
    if (m_fmap[a] == (size_t) -1) {
        m_fmap[a] = a; m_rmap[a] = a; m_ftr[a].reset();
    }

    // If b was forbidden allow it (create a one-loop)
    if (m_fmap[b] == (size_t) -1) {
        m_fmap[b] = b; m_rmap[b] = b; m_ftr[b].reset();
    }

    // check if b is in the same loop as a
    size_t ax = a, axf = m_fmap[ax];
    scalar_transf<T> sx;
    while (ax < axf && ax < b) {
        sx.transform(m_ftr[ax]);
        ax = axf; axf = m_fmap[ax];
    }
    if (ax == b) {
        if (swapped) sx.invert();
        if (sx != tr) {
            throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Mapping exists with different sign.");
        }
        return;
    }

    size_t br = m_rmap[b], bf = m_fmap[b];
    scalar_transf<T> sab(tr); // cur a -> cur b
    if (swapped) sab.invert();
    while (b != bf) {
        // remove b from its loop
        sx = m_ftr[b];
        m_fmap[br] = bf; m_rmap[bf] = br;
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

    static const char *method = "mark_forbidden(const index<N>&)";

#ifdef LIBTENSOR_DEBUG
    if(!is_valid_pidx(idx)) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "idx");
    }
#endif // LIBTENSOR_DEBUG

    abs_index<N> aidx(idx, m_pdims);
    size_t a = aidx.get_abs_index();

    if (m_fmap[a] == (size_t) -1) return;

    size_t af = m_fmap[a];
    while (af != a) {
        size_t ax = af;
        af = m_fmap[ax];
        m_fmap[ax] = (size_t) -1;
        m_rmap[ax] = (size_t) -1;
        m_ftr[ax].reset();
    }
    m_fmap[a] = (size_t) -1;
    m_rmap[a] = (size_t) -1;
    m_ftr[a].reset();
}


template<size_t N, typename T>
index<N> se_part<N, T>::get_direct_map(const index<N> &idx) const {

    static const char *method = "get_direct_map(const index<N>&)";

    abs_index<N> apidx(idx, m_pdims);
#ifdef LIBTENSOR_DEBUG
    if (m_fmap[apidx.get_abs_index()] == (size_t) -1)
        throw bad_parameter(g_ns, k_clazz, method,
                            __FILE__, __LINE__, "Partition is not allowed.");
#endif

    abs_index<N> afpidx(m_fmap[apidx.get_abs_index()], m_pdims);
    return afpidx.get_index();
}


template<size_t N, typename T>
scalar_transf<T> se_part<N, T>::get_transf(const index<N> &from,
        const index<N> &to) const {

    static const char *method = "get_transf(const index<N>&, const index<N>&)";

    size_t a = abs_index<N>(from, m_pdims).get_abs_index();
    size_t b = abs_index<N>(to, m_pdims).get_abs_index();

    if (a == b) return scalar_transf<T>();

    bool swapped = false;
    if (a > b) { swapped = true; std::swap(a, b); }
    size_t x = m_fmap[a];
    scalar_transf<T> tr(m_ftr[a]);
    while (x != b && a < x) {
        tr.transform(m_ftr[x]);
        x = m_fmap[x];
    }
    if (x <= a)
        throw bad_symmetry(g_ns, k_clazz, method,
                           __FILE__, __LINE__, "No mapping.");

    if (swapped) tr.invert();
    return tr;
}


template<size_t N, typename T>
bool se_part<N, T>::map_exists(
    const index<N> &from, const index<N> &to) const {

    size_t a = abs_index<N>(from, m_pdims).get_abs_index();
    size_t b = abs_index<N>(to, m_pdims).get_abs_index();

    if (a > b) std::swap(a, b);
    if (m_fmap[a] == (size_t) -1) return false;
    if (m_fmap[b] == (size_t) -1) return false;

    size_t x = m_fmap[a];

    while (x != b && a < x) {
        x = m_fmap[x];
    }
    return (x == b);
}


template<size_t N, typename T>
bool se_part<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

    return m_bis.equals(bis);
}


template<size_t N, typename T>
bool se_part<N, T>::is_allowed(const index<N> &idx) const {

    index<N> pidx;
    for (register size_t i = 0; i < N; i++) {
        register size_t n = m_bidims[i] / m_pdims[i];
        pidx[i] = idx[i] / n;
    }
    
    return !is_forbidden(pidx);
}


template<size_t N, typename T>
void se_part<N, T>::permute(const permutation<N> &perm) {

    if (perm.is_identity()) return;

    m_bis.permute(perm);
    m_bidims.permute(perm);

    bool affects_map = false;
    for (size_t i = 0; i < N; i++) {
        if (m_pdims[i] != 1 && perm[i] != i) { affects_map = true; break; }
    }

    if (affects_map) {
        dimensions<N> pdims(m_pdims);
        m_pdims.permute(perm);

        size_t mapsz = m_pdims.get_size();

        size_t *fmap = m_fmap; m_fmap = new size_t[mapsz];
        size_t *rmap = m_rmap; m_rmap = new size_t[mapsz];
        scalar_transf<T> *ftr = m_ftr; m_ftr = new scalar_transf<T>[mapsz];
        for (size_t i = 0; i < mapsz; i++) {
            m_fmap[i] = m_rmap[i] = i;
        }

        for (size_t i = 0; i < mapsz; i++) {

            if (fmap[i] <= i) continue;

            abs_index<N> aia(i, pdims);
            index<N> ia(aia.get_index());
            ia.permute(perm);
            abs_index<N> naia(ia, m_pdims);

            if (fmap[i] == (size_t) -1) {
                m_fmap[naia.get_abs_index()] =
                    m_rmap[naia.get_abs_index()] = (size_t) -1;
                continue;
            }

            abs_index<N> aib(fmap[i], pdims);
            index<N> ib(aib.get_index());
            ib.permute(perm);
            abs_index<N> naib(ib, m_pdims);

            add_map(naia.get_index(), naib.get_index(), ftr[i]);
        }

        delete [] fmap; fmap = 0;
        delete [] rmap;    rmap = 0;
        delete [] ftr; ftr = 0;
    }
}


template<size_t N, typename T>
void se_part<N, T>::apply(index<N> &idx, tensor_transf<N, T> &tr) const {

    //  Determine partition index and offset within partition
    //
    index<N> pidx, poff;
    for(register size_t i = 0; i < N; i++) {
        register size_t n = m_bidims[i] / m_pdims[i];
        pidx[i] = idx[i] / n;
        poff[i] = idx[i] % n;
    }

    //  Map the partition index
    //
    abs_index<N> apidx(pidx, m_pdims);
    if (m_fmap[apidx.get_abs_index()] == (size_t) -1) return;

    abs_index<N> apidx_mapped(m_fmap[apidx.get_abs_index()], m_pdims);
    pidx = apidx_mapped.get_index();

    //  Construct a mapped block index
    //
    for(register size_t i = 0; i < N; i++) {
        register size_t n = m_bidims[i] / m_pdims[i];
        idx[i] = pidx[i] * n + poff[i];
    }

    tr.transform(m_ftr[apidx.get_abs_index()]);
}


template<size_t N, typename T>
dimensions<N> se_part<N, T>::make_pdims(const block_index_space<N> &bis,
                                        const mask<N> &msk, size_t npart) {

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
    if (! is_valid_pdims(bis, pdims)) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__, "bis");
    }
#endif

    return pdims;
}


template<size_t N, typename T>
bool se_part<N, T>::is_valid_pdims(
    const block_index_space<N> &bis, const dimensions<N> &d) {

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
    if (a < b) {
        while (af < b && a < af) {
            tx.transform(m_ftr[a]);
            a = af; af = m_fmap[a];
        }
    }
    else {
        while ((af < b && b < a) || (a < af && af < b) || (b < a && a < af)) {
            tx.transform(m_ftr[a]);
            a = af; af = m_fmap[a];
        }
    }
    tx.transform(m_ftr[a]);
    m_fmap[a] = b; m_rmap[b] = a;
    m_fmap[b] = af; m_rmap[af] = b;
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

