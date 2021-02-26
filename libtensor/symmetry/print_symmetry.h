#ifndef LIBTENSOR_PRINT_SYMMETRY_H
#define LIBTENSOR_PRINT_SYMMETRY_H

#include <iomanip>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/print_dimensions.h>
#include <libtensor/core/symmetry.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/symmetry_element_set_adapter.h>

/** \file print_symmetry.h

    Several operators for printing symmetry information.
 **/

namespace libtensor {

//! \name Declaration of symmetry information print routines
//@{

template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os, const symmetry<N, T> &sym);

template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os, const symmetry_element_set<N, T> &sym);

template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os, const se_label<N, T> &se);

template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os, const se_part<N, T> &se);

template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os, const se_perm<N, T> &se);

template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os, const block_labeling<N> &bl);

template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os, const evaluation_rule<N> &er);

template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os, const product_rule<N> &pr);

template<typename T>
std::ostream &operator<<(std::ostream &os, const scalar_transf<T> &tr);


//@}


template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os, const symmetry<N, T> &sym) {

    size_t i = 1;
    for (typename symmetry<N, T>::iterator it = sym.begin();
            it != sym.end(); it++, i++) {

        const symmetry_element_set<N, T> &set = sym.get_subset(it);
        os << " " << std::setw(2) << i << ". " << set << std::endl;
    }
    return os;
}


template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os,
        const symmetry_element_set<N, T> &set) {

    typedef se_label<N, T> se1_t;
    typedef se_part<N, T> se2_t;
    typedef se_perm<N, T> se3_t;
    typedef symmetry_element_set_adapter<N, T, se1_t> adapter1_t;
    typedef symmetry_element_set_adapter<N, T, se2_t> adapter2_t;
    typedef symmetry_element_set_adapter<N, T, se3_t> adapter3_t;

    const std::string &id = set.get_id();
    std::cout << "Set " << id << std::endl;
    if (id.compare(se1_t::k_sym_type) == 0) {
        adapter1_t g(set);
        for (typename adapter1_t::iterator its = g.begin();
                its != g.end(); its++) {
            os << g.get_elem(its);
        }
    }
    else if (id.compare(se2_t::k_sym_type) == 0) {
        adapter2_t g(set);
        for (typename adapter2_t::iterator its = g.begin();
                its != g.end(); its++) {
            os << g.get_elem(its);
        }
    }
    else if (id.compare(se3_t::k_sym_type) == 0) {
        adapter3_t g(set);
        for (typename adapter3_t::iterator its = g.begin();
                its != g.end(); its++) {
            os << g.get_elem(its) << std::endl;
        }
    }
    return os;
}


template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os, const se_label<N, T> &se) {

    os << "Table ID: " << se.get_table_id() << std::endl;
    os << "Block labels: " << se.get_labeling() << std::endl;
    os << "Rule: " << se.get_rule();
    return os;
}

template<size_t N, typename T>
std::ostream &operator<<(std::ostream &os, const se_part<N, T> &se) {

    const dimensions<N> &pdims = se.get_pdims();
    os << "Partition dims: " << pdims << std::endl;
    os << "Mappings:";
    abs_index<N> ai(pdims);
    do {
        if (se.is_forbidden(ai.get_index())) {
            os << std::endl << " " << ai.get_index() << " (x)";
            continue;
        }

        abs_index<N> aix(se.get_direct_map(ai.get_index()), pdims);
        if (aix.get_abs_index() <= ai.get_abs_index()) continue;

        os << std::endl << " " << ai.get_index() << " -> " << aix.get_index();
        os << " (" << se.get_transf(ai.get_index(), aix.get_index()) << ")";
    } while (ai.inc());

    return os;
}

template<size_t N, typename T> 
std::ostream &operator<<(std::ostream &out, const se_perm<N, T> &sp) {

    out << sp.get_perm() << " " << sp.get_transf();
    return out;
}


template<size_t N>
std::ostream &operator<<(std::ostream &os, const block_labeling<N> &bl) {

    typedef product_table_i::label_t label_type;
    for (size_t i = 0; i < N; i++) {
        size_t itype = bl.get_dim_type(i);
        os << " [" << i << "(" << itype << "):";
        for (size_t j = 0; j < bl.get_dim(itype); j++) {
            label_type l = bl.get_label(itype, j);
            if (l == product_table_i::k_invalid) os << " *";
            else os << " " << l;
        }
        os << "]";
    }
    return os;
}

template<size_t N>
std::ostream &operator<<(std::ostream &os, const evaluation_rule<N> &er) {

    for (typename evaluation_rule<N>::iterator it = er.begin();
            it != er.end(); it++) {
        os << " " << er.get_product(it);
    }
    return os;
}

template<size_t N>
std::ostream &operator<<(std::ostream &os, const product_rule<N> &pr) {

    for (typename product_rule<N>::iterator it = pr.begin();
            it != pr.end(); it++) {
        os << "([";
        const sequence<N, size_t> &seq = pr.get_sequence(it);
        for (size_t j = 0; j < N; j++) os << seq[j];
        os << "], ";
        product_table_i::label_t l = pr.get_intrinsic(it);
        if (l == product_table_i::k_invalid) os << "*";
        else os << l;
        os << ")";
    }
    return os;
}

//@}

} // namespace libtensor

#endif // LIBTENSOR_PRINT_SYMMETRY
