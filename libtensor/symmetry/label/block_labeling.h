#ifndef LIBTENSOR_BLOCK_LABELING_H
#define LIBTENSOR_BLOCK_LABELING_H

#include "../../core/dimensions.h"
#include "../../core/mask.h"
#include "../../exception.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Labels of block indexes of a block %tensor.
    \tparam N Block %tensor dimensions.

 **/
template<size_t N>
class block_labeling {
public:
    static const char *k_clazz;

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_group label_group;

private:
    dimensions<N> m_bidims; //!< Block index dimensions
    sequence<N, size_t> m_type; //!< Types of every dimension
    sequence<N, label_group*> m_labels; //!< Label types

public:
    //! \name Constructor/Destructor
    //@{

    /** \brief Creates a class of unassigned block labels

        \param bidims Block index dimensions.
     **/
    block_labeling(const dimensions<N> &bidims);

    /** \brief Copy constructor
     **/
    block_labeling(const block_labeling<N> &bl);

    /** \brief Destructor
     **/
    ~block_labeling();

    //@}

    //! \name Object manipulations
    //@{

    /** \brief Assign label to block indexes given by %mask and position
        \param msk Dimension mask
        \param blk Block position
        \param label Block label
     **/
    void assign(const mask<N> &msk, size_t blk, label_t label);

    /** \brief Remove all labels
     **/
    void clear();

    /** \brief Match labels
     **/
    void match();

    /** \brief Permute the indexes.
     **/
    void permute(const permutation<N> &p);

    //@}

    //! \name Access functions
    //@{

    /** \brief Returns the block index dimensions
     **/
    const dimensions<N> &get_block_index_dims() const {
        return m_bidims;
    }

    /** \brief Returns the type (labeling pattern) of a dimension
        \param dim Dimension number.
        \throw out_of_bounds If the dimension number is out of bounds.
     **/
    size_t get_dim_type(size_t dim) const;

    /** \brief Returns the block dimension of a dimension type
        \param type Dimension type.
        \throw bad_parameter If the dimension type is invalid.
     **/
    size_t get_dim(size_t type) const throw(out_of_bounds);

    /** \brief Returns the label of a block of a dimension type.
        \param type Dimension type.
        \param pos Block position.
        \throw out_of_bounds If the dimension type is out of bounds.
     **/
    label_t get_label(size_t type, size_t pos) const throw(out_of_bounds);

    //@}
};

/** \brief Transfer labels between to block_labeling objects
    \param from Source of labeling
    \param map Index map
    \param to Where to put the labels

    The map specifies how the labeling is to be transfered:
    - Dimension \c i in \c from is transfered to dimension \c map[i] in \c to
    - If \c map[i] is -1 the dimension is ignored.

 **/
template<size_t N, size_t M>
void transfer_labeling(const block_labeling<N> &from,
        const sequence<N, size_t> &map, block_labeling<M> &to);

} // namespace libtensor


#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class block_labeling<1>;
    extern template class block_labeling<2>;
    extern template class block_labeling<3>;
    extern template class block_labeling<4>;
    extern template class block_labeling<5>;
    extern template class block_labeling<6>;

    extern template void transfer_labeling(const block_labeling<1> &,
            const sequence<1, size_t> &, block_labeling<1> &);
    extern template void transfer_labeling(const block_labeling<1> &,
            const sequence<1, size_t> &, block_labeling<2> &);
    extern template void transfer_labeling(const block_labeling<1> &,
            const sequence<1, size_t> &, block_labeling<3> &);
    extern template void transfer_labeling(const block_labeling<1> &,
            const sequence<1, size_t> &, block_labeling<5> &);
    extern template void transfer_labeling(const block_labeling<1> &,
            const sequence<1, size_t> &, block_labeling<6> &);

    extern template void transfer_labeling(const block_labeling<2> &,
            const sequence<2, size_t> &, block_labeling<1> &);
    extern template void transfer_labeling(const block_labeling<2> &,
            const sequence<2, size_t> &, block_labeling<2> &);
    extern template void transfer_labeling(const block_labeling<2> &,
            const sequence<2, size_t> &, block_labeling<3> &);
    extern template void transfer_labeling(const block_labeling<2> &,
            const sequence<2, size_t> &, block_labeling<5> &);
    extern template void transfer_labeling(const block_labeling<2> &,
            const sequence<2, size_t> &, block_labeling<6> &);

    extern template void transfer_labeling(const block_labeling<3> &,
            const sequence<3, size_t> &, block_labeling<1> &);
    extern template void transfer_labeling(const block_labeling<3> &,
            const sequence<3, size_t> &, block_labeling<2> &);
    extern template void transfer_labeling(const block_labeling<3> &,
            const sequence<3, size_t> &, block_labeling<3> &);
    extern template void transfer_labeling(const block_labeling<3> &,
            const sequence<3, size_t> &, block_labeling<5> &);
    extern template void transfer_labeling(const block_labeling<3> &,
            const sequence<3, size_t> &, block_labeling<6> &);

    extern template void transfer_labeling(const block_labeling<4> &,
            const sequence<4, size_t> &, block_labeling<1> &);
    extern template void transfer_labeling(const block_labeling<4> &,
            const sequence<4, size_t> &, block_labeling<2> &);
    extern template void transfer_labeling(const block_labeling<4> &,
            const sequence<4, size_t> &, block_labeling<3> &);
    extern template void transfer_labeling(const block_labeling<4> &,
            const sequence<4, size_t> &, block_labeling<5> &);
    extern template void transfer_labeling(const block_labeling<4> &,
            const sequence<4, size_t> &, block_labeling<6> &);

    extern template void transfer_labeling(const block_labeling<5> &,
            const sequence<5, size_t> &, block_labeling<1> &);
    extern template void transfer_labeling(const block_labeling<5> &,
            const sequence<5, size_t> &, block_labeling<2> &);
    extern template void transfer_labeling(const block_labeling<5> &,
            const sequence<5, size_t> &, block_labeling<3> &);
    extern template void transfer_labeling(const block_labeling<5> &,
            const sequence<5, size_t> &, block_labeling<5> &);
    extern template void transfer_labeling(const block_labeling<5> &,
            const sequence<5, size_t> &, block_labeling<6> &);

    extern template void transfer_labeling(const block_labeling<6> &,
            const sequence<6, size_t> &, block_labeling<1> &);
    extern template void transfer_labeling(const block_labeling<6> &,
            const sequence<6, size_t> &, block_labeling<2> &);
    extern template void transfer_labeling(const block_labeling<6> &,
            const sequence<6, size_t> &, block_labeling<3> &);
    extern template void transfer_labeling(const block_labeling<6> &,
            const sequence<6, size_t> &, block_labeling<5> &);
    extern template void transfer_labeling(const block_labeling<6> &,
            const sequence<6, size_t> &, block_labeling<6> &);

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "block_labeling_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

#endif // LIBTENSOR_BLOCK_LABELING_H

