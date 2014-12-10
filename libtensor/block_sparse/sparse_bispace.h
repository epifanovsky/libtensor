#ifndef SPARSE_BISPACE_H
#define SPARSE_BISPACE_H

#include "sparse_bispace_impl.h"
#include "sparsity_expr.h"
#include <libtensor/core/permutation.h>

namespace libtensor {


template<size_t N>
class sparse_bispace : public sparse_bispace_impl
{
public:
    sparse_bispace(const sparse_bispace_impl& impl) : sparse_bispace_impl(impl) {}
    template<size_t M>
    sparse_bispace<N+M> operator|(const sparse_bispace<M>& rhs);
    sparse_bispace<N> permute(const permutation<N>& perm) const;
};

template<size_t N> template<size_t M>
sparse_bispace<N+M> sparse_bispace<N>::operator|(const sparse_bispace<M>& rhs)
{
    return static_cast< sparse_bispace<N+M> >(sparse_bispace_impl(*this,rhs));
}

template<size_t N> 
sparse_bispace<N> sparse_bispace<N>::permute(const permutation<N>& perm) const
{
    runtime_permutation rperm(N);
    for(size_t i = 0; i < N; ++i) rperm[i] = perm[i]; 
    return static_cast< sparse_bispace<N> >(sparse_bispace_impl::permute(rperm));
}

template<>
class sparse_bispace<1> : public sparse_bispace_impl
{
public:
    sparse_bispace(size_t dim,
                   const idx_list& split_points = idx_list(1,0)) : sparse_bispace_impl(std::vector<subspace>(1,subspace(dim,split_points))) {}

    void split(const idx_list& split_points) { m_subspaces[0].split(split_points); }
    size_t get_dim() const { return m_subspaces[0].get_dim(); }

    template<size_t M>
    sparse_bispace<M+1> operator|(const sparse_bispace<M>& rhs);

    sparsity_expr<1,1> operator%(const sparse_bispace<1>& rhs) const;

};

template<size_t M>
sparse_bispace<M+1> sparse_bispace<1>::operator|(const sparse_bispace<M>& rhs)
{
    return static_cast< sparse_bispace<M+1> >(sparse_bispace_impl(*this,rhs));
}

inline sparsity_expr<1,1> sparse_bispace<1>::operator%(const sparse_bispace<1>& rhs) const
{
    return sparsity_expr<1,1>(*this,rhs);
}

//Implementation of methods in sparsity_expr and sparse_block_tree requiring sparse_bispace definition
    
//Internal method for recursively constructing a list of all 1d bispaces used to create this expr
template<size_t M>
void sparsity_expr<M,1>::retrieve_subspaces(std::deque<subspace>& subspaces) const
{
    subspaces.push_front(m_cur_bispace.m_subspaces[0]);
}
    
template<size_t M,size_t N>
void sparsity_expr<M,N>::retrieve_subspaces(std::deque<subspace>& subspaces) const
{
    subspaces.push_front(m_cur_bispace.m_subspaces[0]);
    m_sub_expr.retrieve_subspaces(subspaces);
}
    
template<size_t M>
sparse_bispace<2> sparsity_expr<M,1>::operator<<(const std::vector< sequence<2,size_t> >& sig_blocks)
{
    std::vector<idx_list> keys;
    for(size_t i = 0; i < sig_blocks.size(); ++i)
    {
        idx_list ent; 
        ent.push_back(sig_blocks[i][0]);
        ent.push_back(sig_blocks[i][1]);
        keys.push_back(ent);
    }
    std::vector<subspace> subs(m_parent_bispace.m_subspaces);
    subs.insert(subs.end(),m_cur_bispace.m_subspaces.begin(),m_cur_bispace.m_subspaces.end());
    return static_cast< sparse_bispace<2> >(sparse_bispace_impl(
                                    subs,
                                    std::vector<sparsity_data>(1,sparsity_data(2,keys)),
                                    idx_list(1,0)));
}

template<size_t M,size_t N>
sparse_bispace<M+N> sparsity_expr<M,N>::operator<<(const std::vector< sequence<N+1,size_t> >& sig_blocks)
{

    std::deque<subspace> subspaces;
    retrieve_subspaces(subspaces);

    std::vector<idx_list> keys;
    for(size_t i = 0; i < sig_blocks.size(); ++i)
    {
        idx_list ent; 
        for(size_t j = 0; j < N+1; ++j)
            ent.push_back(sig_blocks[i][j]);
        keys.push_back(ent);
    }
    std::vector<subspace> subs(m_parent_bispace.m_subspaces);
    subs.insert(subs.end(),subspaces.begin(),subspaces.end());
    return static_cast< sparse_bispace<M+N> >(sparse_bispace_impl(
                                    subs,
                                    std::vector<sparsity_data>(1,sparsity_data(N+1,keys)),
                                    idx_list(1,0)));
}

} //namespace libtensor

#if 0
//Useful types
typedef std::vector<size_t> block_list;

//Forward declarations to dodge 'specialization after instantiation' compiler errors 
template<size_t N>
class sparse_bispace; 

//Utility functions

/**  One-dimensional sparse block index space
 **/
template<>
class sparse_bispace<1> {
private:
    size_t m_dim; //!< Number of elements
    std::vector<size_t> m_abs_indices; //!< Block absolute starting indices
    
    //Constructor used to instantiate 1d bispaces via the contraction of an index in a 2d bispace
    //Explicitly need this constructor because 1d sparse_block_trees do not implement contract() method
    //Implemented below to avoid incomplete type errors
    sparse_bispace(const sparse_bispace<2>& parent,size_t contract_idx);
public: 

    /** \brief Creates the sparse block %index space with a given dimension
        \param dim Number of elements in this space.
     **/
    explicit sparse_bispace(size_t dim);
    
    /** \brief Returns the dimension of the block index space 
     **/
    size_t get_dim() const;

    size_t get_nnz() const { return m_dim; }

    /** \brief Returns the number blocks into which this space has been split 
     **/
    size_t get_n_blocks() const;

    /** \brief Splits this space into blocks with offsets starting at offsets
               in split_points. First block always starts at zero
        \param split_points Iterable container of absolute indices where each block should start 
        \throw out_of_bounds If a split_point value exceeds the index limits, or if a zero length vector is passed 
     **/
    void split(const std::vector<size_t> &split_points)
        throw(out_of_bounds);

    /** \brief Returns the size of the block with block index block_idx
        \throw out_of_bounds If (# of blocks  - 1) < block_idx || block_idx < 0
     **/
    size_t get_block_size(size_t block_idx) const 
        throw(out_of_bounds);

    /** \brief Returns the absolute starting index of the block with block index block_idx
        \throw out_of_bounds If (# of blocks  - 1) < block_idx < 0
     **/
    size_t get_block_abs_index(size_t block_idx) const 
        throw(out_of_bounds);

    /** \brief Returns a N+1 d sparse bispace
     *         Called during resolution of sparsity expressions
     **/
    template<size_t M>
    sparse_bispace<M+1> operator|(const sparse_bispace<M>& rhs) const;

    /** \brief Returns a sparsity_expr corresponding to a 2d bispace 
     **/
    sparsity_expr<1,1> operator%(const sparse_bispace<1>& rhs) const;

    /** \brief Returns a copy of this object 
        \throw out_of_bounds If an inappropriate index is specified 
     **/
    const sparse_bispace<1>& operator[](size_t  idx) const
        throw(out_of_bounds);

    /** Stub methods for general compatibility, even though can't have sparse groups in a 1d bispace
     **/
    size_t get_n_sparse_groups() const { return 0; }

    const sparse_block_tree& get_sparse_group_tree(size_t group_idx) const { throw bad_parameter(g_ns,"sparse_bispace<1>","get_sparse_group_tree(...)",__FILE__,__LINE__,"not implemented"); }
    size_t get_sparse_group_offset(size_t group_idx) const { throw bad_parameter(g_ns,"sparse_bispace<1>","get_sparse_group_tree(...)",__FILE__,__LINE__,"not implemented"); }

    size_t get_n_index_groups() const { return 1; }
    size_t get_index_group_offset(size_t grp_idx) const { return 0; }
    size_t get_index_group_order(size_t grp_idx) const { return 1; }
    size_t get_index_group_dim(size_t grp_idx) const { return m_dim; }
    size_t get_index_group_containing_subspace(size_t subspace_idx) const { if(subspace_idx != 0) {  throw bad_parameter(g_ns,"sparse_bispace<1>","get_index_group_containing_subspace",__FILE__,__LINE__,"only one subspace"); }  return 0; }

    void truncate_subspace(size_t subspace_idx,const idx_pair& bounds);

    std::vector<idx_pair> get_batches(size_t subspace_idx,size_t max_n_elem) const;
    size_t get_batch_size(size_t subspace_idx,const idx_pair& batch) const;

    /** \brief Returns whether this object is equal to another. 
     *         Equality is defined to be the same dimension and block splitting pattern
     **/
    bool operator==(const sparse_bispace<1>& rhs) const;

    /** \brief Returns whether this object is not equal to another. 
     *         Equality is defined to be the same dimension and block splitting pattern
     **/
    bool operator!=(const sparse_bispace<1>& rhs) const;
};


inline sparse_bispace<1>::sparse_bispace(size_t dim) : m_dim(dim)
{ 
    m_abs_indices.push_back(0);
} 

inline size_t sparse_bispace<1>::get_dim() const
{
    return m_dim;
}

inline size_t sparse_bispace<1>::get_n_blocks() const
{
    return m_abs_indices.size();
}

inline void sparse_bispace<1>::split(const std::vector<size_t> &split_points) throw(out_of_bounds)
{
    if(split_points.size() < 1 || split_points.size() > (m_dim - 1))
    {
        throw out_of_bounds(g_ns,"sparse_bispace<1>","split(...)",
                __FILE__,__LINE__,"Must have 1 <= # of split points <= dim - 1"); 
    }

    for(int i = 0; i < split_points.size(); ++i)
    {
        size_t split_point = split_points[i];
        if(split_point > (m_dim - 1))
        {
            throw out_of_bounds(g_ns,"sparse_bispace<1>","split(...)",
                    __FILE__,__LINE__,"Split point indices cannot exceed (dim - 1)"); 
        }
        else if(split_point <= m_abs_indices.back())
        {
            throw out_of_bounds(g_ns,"sparse_bispace<1>","split(...)",
                    __FILE__,__LINE__,"Split point indices must be strictly increasing"); 
        }
        m_abs_indices.push_back(split_point);
    }
}

inline size_t sparse_bispace<1>::get_block_size(size_t block_idx) const throw(out_of_bounds)
{
    if(block_idx > (m_abs_indices.size() - 1))
    {
        throw out_of_bounds(g_ns,"sparse_bispace<1>","get_block_size(size_t block_idx)",
                __FILE__,__LINE__,"Cannot pass block_idx greater than (# of blocks - 1)"); 
    }
    else if(block_idx == (m_abs_indices.size() - 1))
    {
        return m_dim - m_abs_indices.back(); 
    }
    else
    {
        return m_abs_indices[block_idx + 1] - m_abs_indices[block_idx];
    }
}


inline size_t sparse_bispace<1>::get_block_abs_index(size_t block_idx) const throw(out_of_bounds)
{
    if(block_idx > (m_abs_indices.size() - 1))
    {
        throw out_of_bounds(g_ns,"sparse_bispace<1>","get_block_abs_index(size_t block_idx)",
                __FILE__,__LINE__,"Cannot pass block_idx greater than (# of blocks - 1)"); 
    }
    return m_abs_indices[block_idx];
}

    /** \brief Returns a copy of this object 
        \throw out_of_bounds If an inappropriate index is specified 
     **/
inline const sparse_bispace<1>& sparse_bispace<1>::operator[](size_t idx) const throw(out_of_bounds)
{
    if(idx != 0)
    {
        throw out_of_bounds(g_ns,"sparse_bispace<1>","operator[](...)",
                __FILE__,__LINE__,"Invalid subspace index specified (can only specify 0"); 
    }
    return *this;
}

inline void sparse_bispace<1>::truncate_subspace(size_t subspace_idx, const idx_pair& bounds)
{
    if(subspace_idx != 0)
    {
        throw out_of_bounds(g_ns,"sparse_bispace<1>","truncate_subspace(...)",
                __FILE__,__LINE__,"Invalid subspace index specified (can only specify 0"); 

    }

    //Construct new dim and split points
    size_t dim = 0;
    std::vector<size_t> split_points;
    for(size_t block_idx = bounds.first+1; block_idx < bounds.second; ++block_idx)
    {
        split_points.push_back(m_abs_indices[block_idx] - m_abs_indices[bounds.first]);
        dim += m_abs_indices[block_idx] - m_abs_indices[block_idx-1];
    }
    size_t upper_bound = (bounds.second == m_abs_indices.size()) ? m_dim : m_abs_indices[bounds.second];
    dim += upper_bound - m_abs_indices[bounds.second - 1];

    sparse_bispace<1> truncated(dim);
    if(split_points.size() > 0)
    {
        truncated.split(split_points);
    }
    *this = truncated;
}

inline std::vector<idx_pair> sparse_bispace<1>::get_batches(size_t subspace_idx,size_t max_n_elem) const
{
    if(subspace_idx != 0)
    {
        throw out_of_bounds(g_ns,"sparse_bispace<1>","get_batches(...)",
                __FILE__,__LINE__,"Can only return batches for subspace 0"); 
    }

    std::vector<idx_pair> batches;
    size_t start_idx = 0;
    size_t n_elem = 0;
    for(size_t block_idx = 0; block_idx < get_n_blocks(); ++block_idx)
    {
        size_t this_block_contrib = get_block_size(block_idx);
        if(n_elem + this_block_contrib > max_n_elem)
        {
            batches.push_back(idx_pair(start_idx,block_idx));
            start_idx = block_idx;
            n_elem = 0;
        }
        n_elem += this_block_contrib;
    }
    batches.push_back(idx_pair(start_idx,get_n_blocks()));
    return batches;
}

inline size_t sparse_bispace<1>::get_batch_size(size_t subspace_idx,const idx_pair& batch) const
{
    size_t batch_size = 0;
    for(size_t block_idx = batch.first; block_idx < batch.second; ++block_idx)
    {
        batch_size += get_block_size(block_idx);
    }
    return batch_size;
}

inline bool sparse_bispace<1>::operator==(const sparse_bispace<1>& rhs) const
{
    return (this->m_dim == rhs.m_dim) && (this->m_abs_indices == rhs.m_abs_indices);
}

inline bool sparse_bispace<1>::operator!=(const sparse_bispace<1>& rhs) const
{
    return ! (*this == rhs);
} 

//General N-dimensional case
template<size_t N>
class sparse_bispace {
public: 
    static const char *k_clazz; //!< Class name
private:

    std::vector< sparse_bispace<1> > m_subspaces;

    //Contains the indices in m_subspaces at which each group of sparse coupled indices starts
    std::vector<size_t> m_sparse_indices_sets_offsets;

    //Contains the trees that describe the sparsity of each group of coupled indices  
    std::vector< sparse_block_tree > m_sparse_block_trees;

    //Internal-use array containing the dimension of each subspace/sparse composite subspace group. 
    //Used to calculate number of elements and offsets
    //Should never be edited directly - instead call init()
    std::vector<size_t> m_index_group_dims;

    //Internal use array for improving performance by pre-computing the inner size of each index
    std::vector<size_t> m_inner_sizes;

    //Internal array used for the purpose of grouping coupled subspaces together 
    std::vector<size_t> m_index_group_offsets;
    
    //Used by get_block_offset to lookup offsets in sparse trees
    //Initialized only once by init() for performance
    std::vector< std::vector<size_t> > m_sparse_key_vecs;
    
    //Helper functions used to set the m_index_group_dims array used for calculating block offsets and nnz
    //Also initializes other sparsity data
    void init();
    
    //Used to initialize offsets stored in sparse_block_tree members
    size_t set_offsets(sparse_block_tree& tree,const std::vector<size_t>& positions);

    /** \brief Constructors a composite sparse_bispace from two component spaces
     *         Used to implement operator|(...) between multi-dimensional spaces
     **/
    template<size_t L> 
    sparse_bispace(const sparse_bispace<N-L>& lhs,const sparse_bispace<L>& rhs);

    /** \brief Constructor used by the following operator pattern:
     *         <N> = <L> & <N - L> << { <N - L + 1> }
     **/
    template<size_t L>
    sparse_bispace(const sparse_bispace<N-L+1>& lhs,const std::vector< sparse_bispace<1> >& rhs_subspaces,const std::vector< sequence<L,size_t> >& sig_blocks);

    /** Used by contract() to produce lower-rank bispace
     **/
    sparse_bispace(const sparse_bispace<N+1>& parent,size_t contract_idx);   
    
    /** Used by fuse() to produce higher-rank bispace
     **/
    template<size_t L> 
    sparse_bispace(const sparse_bispace<N-L+1>& lhs, const sparse_bispace<L>& rhs);

    //Worker function used in implementing constructors
    //Special case for when RHS has no sparse members bcs 1d
    //Offset parameter does nothing, needed for proper overloading
    void absorb_sparsity(const sparse_bispace<1>& rhs,size_t offset=0); 

    //General case
    //offset specifies how much to shift the location of the specified sparse indices sets
    template<size_t L>
    void absorb_sparsity(const sparse_bispace<L>& rhs,size_t offset=0); 
public:

    /** \brief Returns the number of non-zero elements in this sparse bispace
     **/
    size_t get_nnz() const;

    /** \brief Combines the two operands to produce a new space
     **/
    template<size_t M> 
    sparse_bispace<N+M> operator|(const sparse_bispace<M>& rhs);

    /** \brief Retrieves the appropriate index subspace of this multidimensional space
        \throw out_of_bounds If an inappropriate index is specified 
     **/
    const sparse_bispace<1>& operator[](size_t  idx) const
        throw(out_of_bounds);

    /** \brief Returns an appropriately permuted copy of this bispace 
     **/
    sparse_bispace<N> permute(const permutation<N>& perm) const; 

    /** \brief Returns the bispace resulting from the removal of a particular subspace and the
     *         subsequent aggregation of any sparsity involving that space
     **/
    sparse_bispace<N-1> contract(size_t contract_idx) const;

    void truncate_subspace(size_t subspace_idx,const idx_pair& bounds);
    std::vector<idx_pair> get_batches(size_t subspace_idx,size_t max_n_elem) const;
    size_t get_batch_size(size_t subspace_idx,const idx_pair& batch) const;

    template<size_t L>
    sparse_bispace<N+L-1> fuse(const sparse_bispace<L>& rhs) const;

    /** \brief Returns the number of sparse index groups (0 for fully dense)
     **/
    size_t get_n_sparse_groups() const { return m_sparse_block_trees.size(); }

    /** \brief Access the tree corresponding to sparse index group group_idx
     *         Two N-D spaces are equal if all of their subspaces are equal and in the same order
     **/
    const sparse_block_tree& get_sparse_group_tree(size_t group_idx) const;

    /** \brief Get the subspace index corresponding to the beginning of a given sparsity coupled index group 
     **/
    size_t get_sparse_group_offset(size_t group_idx) const;

    size_t get_n_index_groups() const { return m_index_group_dims.size(); }
    size_t get_index_group_offset(size_t grp_idx) const { return m_index_group_offsets[grp_idx]; }
    size_t get_index_group_dim(size_t grp_idx) const { return m_index_group_dims[grp_idx]; }
    size_t get_index_group_order(size_t grp_idx) const;
    size_t get_index_group_containing_subspace(size_t subspace_idx) const;

    /** \brief Returns whether this object is equal to another of the same dimension. 
     *         Two N-D spaces are equal if all of their subspaces are equal and in the same order  
     **/
    bool operator==(const sparse_bispace<N>& rhs) const;

    /** \brief Returns whether this object is not equal to another of the same dimension. 
     *         Two N-D spaces are equal if all of their subspaces are equal and in the same order
     **/
    bool operator!=(const sparse_bispace<N>& rhs) const;



    //Friend all other types of sparse_bispaces to allow for creation of larger ones from smaller ones
    template<size_t M>
    friend class sparse_bispace;

    //Friend sparsity expr so it can build bispaces when expression is evaluated
    template<size_t L,size_t M>
    friend class sparsity_expr;
};

//Determines the dimensions of each sparsity group in the bispace
//bispaces not belonging to a group simply contribute their dense dimension
template<size_t N>
void sparse_bispace<N>::init()
{
    //Assume init() has been called before
    m_index_group_offsets.resize(0);
    m_index_group_dims.resize(0);
    m_sparse_key_vecs.resize(0);

    size_t subspace_idx = 0; 
    size_t cur_group_idx = 0;
    while(subspace_idx < N)
    {
        m_index_group_offsets.push_back(subspace_idx);

        //Anything sparse in this bispace?
        bool treat_as_sparse = false;
        if(cur_group_idx < m_sparse_indices_sets_offsets.size())
        {
            //Are we in a sparse group?
            if(subspace_idx == m_sparse_indices_sets_offsets[cur_group_idx])
            {
                treat_as_sparse = true;
            }
        }

        if(treat_as_sparse)
        {
            //We are in a sparse group, use the total group size
            m_index_group_dims.push_back(m_sparse_block_trees[cur_group_idx].get_nnz());
            subspace_idx += m_sparse_block_trees[cur_group_idx].get_order();
            ++cur_group_idx;
        }
        else
        {
            m_index_group_dims.push_back(m_subspaces[subspace_idx].get_dim());
            ++subspace_idx;
        }
    }
    
    //Initialize the sparse key vectors for each tree only once to save performance
    for(size_t group_idx = 0; group_idx < m_sparse_block_trees.size(); ++group_idx)
    {
        const sparse_block_tree& sbt = m_sparse_block_trees[group_idx];
        size_t cur_order = sbt.get_order();
        m_sparse_key_vecs.push_back(std::vector<size_t>(cur_order));
    }
    
    //Precompute inner sizes
    m_inner_sizes.resize(m_index_group_dims.size());
    for(size_t inner_size_idx = 0; inner_size_idx < m_index_group_dims.size(); ++inner_size_idx)
    {
        size_t inner_size = 1;
        for(size_t factor_idx = inner_size_idx+1; factor_idx < m_index_group_dims.size(); ++factor_idx)
        {
            inner_size *= m_index_group_dims[factor_idx];
        }
        m_inner_sizes[inner_size_idx] = inner_size;
    }
}

//Worker function used in implementing constructors
//Special case for when RHS has no sparse members bcs 1d
template<size_t N>
void sparse_bispace<N>::absorb_sparsity(const sparse_bispace<1>& rhs,size_t offset)
{
}

//General case
template<size_t N> template<size_t L>
void sparse_bispace<N>::absorb_sparsity(const sparse_bispace<L>& rhs,size_t offset)
{
    //Copy sparsity
    for(size_t i = 0; i < rhs.m_sparse_indices_sets_offsets.size(); ++i)
    {
        m_sparse_indices_sets_offsets.push_back(rhs.m_sparse_indices_sets_offsets[i]+offset);
        m_sparse_block_trees.push_back(rhs.m_sparse_block_trees[i]);
    }
}

//Constructor used by the following pattern: 
//  <N> = <N-L> | <L> pattern
//
template<size_t N> template<size_t L> 
sparse_bispace<N>::sparse_bispace(const sparse_bispace<N-L>& lhs,const sparse_bispace<L>& rhs)
{
    //Copy subspaces
    m_subspaces.reserve(N);
    for(size_t i = 0; i < N-L; ++i)
    {
        m_subspaces.push_back(lhs[i]);
    }
    for(size_t i = 0; i < L; ++i)
    {
        m_subspaces.push_back(rhs[i]);
    }

    absorb_sparsity(lhs);
    absorb_sparsity(rhs,N-L);

    init();
}

//Constructor used by the following operator pattern:
//  <N> = <L> & <N - L> << { <N - L + 1> }
template<size_t N> template<size_t L>
sparse_bispace<N>::sparse_bispace(const sparse_bispace<N-L+1>& lhs,const std::vector< sparse_bispace<1> >& rhs_subspaces,const std::vector< sequence<L,size_t> >& sig_blocks)
{
    //Copy subspaces
    for(size_t i = 0; i < (N-L+1); ++i)
    {
        m_subspaces.push_back(lhs[i]);
    }

    for(size_t i = 0; i < (L-1); ++i)
    {
        m_subspaces.push_back(rhs_subspaces[i]);
    }

    //Initialize the new sparse_block_tree with offset information  
    sparse_block_tree sbt(sig_blocks,m_subspaces);
    std::vector<size_t>  positions(L);
    for(size_t i = 0; i < L; ++i)
    {
        positions[i] = N-L+i;
    }

    m_sparse_block_trees.push_back(sbt);
    m_sparse_indices_sets_offsets.push_back(N-L);

    init();
}


template<size_t N>
size_t sparse_bispace<N>::get_nnz() const
{
    //TODO: Case where all elements are zero, fully sparse tensors????
    size_t nnz = 1;
    for(int i = 0; i < m_index_group_dims.size(); ++i)
    {
        nnz *= m_index_group_dims[i];
    }
    return nnz;
}

template<size_t N> template<size_t M> 
sparse_bispace<N+M> sparse_bispace<N>::operator|(const sparse_bispace<M>& rhs)
{
    return sparse_bispace<N+M>(*this,rhs);
}

//TODO: Should make these check (N-1) instead of m_subspaces.size()
template<size_t N>
const sparse_bispace<1>& sparse_bispace<N>::operator[](size_t idx) const throw(out_of_bounds)
{
    if(idx > (m_subspaces.size() - 1))
    {
        throw out_of_bounds(g_ns,"sparse_bispace<N>","operator[](...)",
                __FILE__,__LINE__,"idx > (# of subspaces - 1) was specified"); 
    }
    return m_subspaces[idx];
}

//TODO: This is haxx, needs to be formally verified
//TODO: This will break if I relocate an entire tree!!!!
template<size_t N> 
sparse_bispace<N> sparse_bispace<N>::permute(const permutation<N>& perm) const
{
    //Sparse metadata is rebuilt as we go
    sparse_bispace<N> copy(*this);
    copy.m_sparse_indices_sets_offsets.clear();

    //Permute subspaces
    for(size_t i = 0; i < N; ++i)
    {
        copy.m_subspaces[i] = m_subspaces[perm[i]];
    }

    //Permute trees
    std::vector<size_t> cur_perm_entries;
    std::vector< sparse_bispace<1> > cur_tree_subspaces;
    size_t cur_tree_idx;
    size_t cur_order;
    size_t cur_tree_start_dest_sub_idx;
    std::map<size_t,size_t> cur_dense_subspaces;
    for(size_t dest_sub_idx = 0; dest_sub_idx < N; ++dest_sub_idx)
    {
        size_t src_sub_idx = perm[dest_sub_idx];

        //Is this index associated with a sparse group?
        bool sparse = false;
        for(size_t sparse_grp_idx = 0; sparse_grp_idx < m_sparse_indices_sets_offsets.size(); ++sparse_grp_idx)
        {
            size_t offset = m_sparse_indices_sets_offsets[sparse_grp_idx];
            size_t order = m_sparse_block_trees[sparse_grp_idx].get_order();
            if((offset <= src_sub_idx) && (src_sub_idx < offset+order))
            {
                //This index comes from this sparse group in the unpermuted bispace
                if(cur_perm_entries.size() == 0)
                {
                    //This is the first index in our tree in the new,permuted,bispace
                    cur_tree_idx = sparse_grp_idx;
                    cur_order = order;
                    cur_tree_start_dest_sub_idx = dest_sub_idx;
                }
                else if(sparse_grp_idx != cur_tree_idx)
                {
                    //This index comes from a different tree, and we haven't filled in our original tree yet
                    throw bad_parameter(g_ns,"sparse_bispace<N>","permute(...)",
                        __FILE__,__LINE__,"permuting between different sparse groups is not supported"); 
                }

                cur_tree_subspaces.push_back(m_subspaces[src_sub_idx]);
                cur_perm_entries.push_back(src_sub_idx - offset);
                sparse = true;
                break;
            }
        }

        if(cur_perm_entries.size() > 0)
        {
            //Did this iteration fill in our tree in the destination bispace?
            if(cur_perm_entries.size() == cur_order)
            {
                //Insert all dense subspaces caught inside this tree
                for(std::map<size_t,size_t>::iterator it = cur_dense_subspaces.begin(); it != cur_dense_subspaces.end(); ++it)
                {
                    size_t cur_dest_sub_idx = it->first;
                    size_t cur_src_sub_idx = it->second;
                    for(size_t entry_idx = 0; entry_idx < cur_perm_entries.size(); ++entry_idx)
                    {
                        if(cur_perm_entries[entry_idx] >= cur_dest_sub_idx)
                        {
                            ++cur_perm_entries[entry_idx];
                        }
                    }
                    cur_perm_entries.insert(cur_perm_entries.begin()+cur_dest_sub_idx,cur_dest_sub_idx);
                    copy.m_sparse_block_trees[cur_tree_idx] = copy.m_sparse_block_trees[cur_tree_idx].insert_subspace(cur_dest_sub_idx - cur_tree_start_dest_sub_idx,m_subspaces[cur_src_sub_idx]);
                }
                runtime_permutation tree_perm(cur_perm_entries);

                //Don't permute if identity
                if(tree_perm != runtime_permutation(cur_order))
                {
                    copy.m_sparse_block_trees[cur_tree_idx] = copy.m_sparse_block_trees[cur_tree_idx].permute(tree_perm);
                    copy.m_sparse_block_trees[cur_tree_idx].set_offsets_sizes_nnz(cur_tree_subspaces);
                }
                copy.m_sparse_indices_sets_offsets.push_back(cur_tree_start_dest_sub_idx);

                cur_dense_subspaces.clear();
                cur_perm_entries.clear();
                cur_tree_subspaces.clear();
            }
            else if(!sparse)
            {
                //We don't want to log irrelevant dense bispaces - only ones caught between sparse bispaces
                cur_dense_subspaces[dest_sub_idx - cur_tree_start_dest_sub_idx] = src_sub_idx;
                cur_tree_subspaces.push_back(m_subspaces[src_sub_idx]);
            }
        }
    }

    copy.init();
    return copy;
}

template<size_t N>
sparse_bispace<N>::sparse_bispace(const sparse_bispace<N+1>& parent,size_t contract_idx) 
{

    //Extract all relevant subspaces
    for(size_t i = 0; i < N+1; ++i)
    {
        if(i == contract_idx)
        {
            continue;
        }
        m_subspaces.push_back(parent.m_subspaces[i]);
    }

    //Contract sparse information appropriately, by default just copying unaffected trees
    for(size_t group_idx = 0; group_idx < parent.m_sparse_block_trees.size(); ++group_idx)
    {
        const sparse_block_tree& cur_tree = parent.m_sparse_block_trees[group_idx];
        size_t offset =   parent.m_sparse_indices_sets_offsets[group_idx];
        size_t order = cur_tree.get_order();

        //Does contraction shift the offset of this tree?
        size_t new_group_offset = offset > contract_idx ? offset-1 : offset;
        
        //Are we contracting this tree?
        if((offset <= contract_idx) && (contract_idx < offset+order))
        {
            //Will contraction wipe out this sparsity?
            if(order == 2)
            {
                continue;
            }
            else
            {

                //What are the subspaces relevant to this tree?
                std::vector< sparse_bispace<1> > tree_subspaces;
                std::vector<size_t> positions;
                for(size_t subspace_idx = new_group_offset; subspace_idx < new_group_offset+order-1; ++subspace_idx)
                {
                    tree_subspaces.push_back(m_subspaces[subspace_idx]);
                    positions.push_back(subspace_idx);
                }

                //Tree-relative idx
                size_t rel_idx = contract_idx - offset;
                sparse_block_tree new_tree = cur_tree.contract(rel_idx,tree_subspaces);
                m_sparse_block_trees.push_back(new_tree);
                m_sparse_indices_sets_offsets.push_back(new_group_offset);
            }
        }
        else
        {
            m_sparse_indices_sets_offsets.push_back(new_group_offset);
            m_sparse_block_trees.push_back(cur_tree);
        }
    }

    init();
}

template<size_t N>
sparse_bispace<N-1> sparse_bispace<N>::contract(size_t contract_idx) const
{
    if(contract_idx > N-1)
    {
        throw bad_parameter(g_ns,"sparse_bispace<N>","contract(...)",
            __FILE__,__LINE__,"contraction index too large"); 
    }
    return sparse_bispace<N-1>(*this,contract_idx);
}

template<size_t N>
void sparse_bispace<N>::truncate_subspace(size_t subspace_idx,const idx_pair& bounds)
{
    //Modify the subspace appropriately
    m_subspaces[subspace_idx].truncate_subspace(0,bounds);

    //Is the subspace in a sparse group? If so, truncate the tree
    size_t target_sparse_grp_idx;
    bool found = false;
    for(size_t sparse_grp_idx = 0; sparse_grp_idx < m_sparse_indices_sets_offsets.size(); ++sparse_grp_idx)
    {
        size_t min = m_sparse_indices_sets_offsets[sparse_grp_idx];
        size_t max = min + m_sparse_block_trees[sparse_grp_idx].get_order();
        if((min <= subspace_idx) && (subspace_idx < max))
        {
            found = true;
            target_sparse_grp_idx = sparse_grp_idx;
            break;
        }
    }
    if(found)
    {
        size_t grp_offset = m_sparse_indices_sets_offsets[target_sparse_grp_idx];
        size_t tree_subspace = subspace_idx - grp_offset;
        sparse_block_tree& cur_tree = m_sparse_block_trees[target_sparse_grp_idx];
        cur_tree = cur_tree.truncate_subspace(tree_subspace,bounds);
        std::vector< sparse_bispace<1> > subspaces;

        //sparse_block_tree::truncate_subspace is designed to preserve the original keys
        //so that we can use them to match up dense offsets in other tensors
        //We need to make the keys internally consistent
        std::vector<std::vector<size_t> > new_keys;
        for(sparse_block_tree::iterator it = cur_tree.begin(); it != cur_tree.end(); ++it)
        {
            std::vector<size_t> cur_key = it.key();
            cur_key[tree_subspace] -= bounds.first;
            new_keys.push_back(cur_key);
        }
        cur_tree = sparse_block_tree(new_keys,cur_tree.get_order());

        for(size_t tree_subspace_idx = 0; tree_subspace_idx < cur_tree.get_order(); ++tree_subspace_idx)
        {
            subspaces.push_back(m_subspaces[grp_offset + tree_subspace_idx]);
        }
        cur_tree.set_offsets_sizes_nnz(subspaces);
    }
    init();
}

template<size_t N>
std::vector<idx_pair> sparse_bispace<N>::get_batches(size_t subspace_idx,size_t max_n_elem) const
{
    bool is_sparse = false;
    size_t idx_grp = get_index_group_containing_subspace(subspace_idx);
    size_t target_sparse_grp;
    size_t idx_grp_offset = get_index_group_offset(idx_grp);
    std::vector<size_t>::const_iterator sparse_pos = find(m_sparse_indices_sets_offsets.begin(),m_sparse_indices_sets_offsets.end(),idx_grp_offset);
    if(sparse_pos != m_sparse_indices_sets_offsets.end())
    {
        is_sparse = true;
        target_sparse_grp = distance(m_sparse_indices_sets_offsets.begin(),sparse_pos);
    }

    std::vector<idx_pair> batches;

    size_t scale_fac = 1;
    for(size_t outer_idx_grp = 0; outer_idx_grp < idx_grp; ++outer_idx_grp)
    {
        scale_fac *= get_index_group_dim(outer_idx_grp);
    }
    for(size_t inner_idx_grp = idx_grp+1; inner_idx_grp < get_n_index_groups(); ++inner_idx_grp)
    {
        scale_fac *= get_index_group_dim(inner_idx_grp);
    }

    size_t start_idx = 0;
    size_t n_elem = 0;
    size_t n_blocks = m_subspaces[subspace_idx].get_n_blocks();
    if(!is_sparse)
    {
        //Dense case
        for(size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
        {
            size_t this_block_contrib = m_subspaces[subspace_idx].get_block_size(block_idx)*scale_fac;
            if(this_block_contrib > max_n_elem)
            {
                throw bad_parameter(g_ns,"sparse_bispace<N>","get_batches(...)",__FILE__,__LINE__,
                        "single block does not fit in batch"); 
                    
            }

            if(n_elem + this_block_contrib > max_n_elem)
            {
                batches.push_back(idx_pair(start_idx,block_idx));
                start_idx = block_idx;
                n_elem = 0;
            }
            n_elem += this_block_contrib;
        }
    }
    else
    {
        //Sparse case
        size_t min = m_sparse_indices_sets_offsets[target_sparse_grp];
        size_t tree_subspace = subspace_idx - min;

        //We must place the batched index at 0 to ensure that it is strictly increasing so that we can batch over it
        const sparse_block_tree& orig_tree = m_sparse_block_trees[target_sparse_grp];
        runtime_permutation perm(orig_tree.get_order());
        perm.permute(0,tree_subspace);
        sparse_block_tree permuted_tree = orig_tree.permute(perm);
        size_t prev_block_idx = 0;
        size_t this_block_idx_subtotal = 0;
        for(sparse_block_tree::iterator it = permuted_tree.begin(); it != permuted_tree.end(); ++it)
        {
            size_t this_block_contrib = (*it)[0].second*scale_fac;
            if(this_block_contrib > max_n_elem)
            {
                throw bad_parameter(g_ns,"sparse_bispace<N>","get_batches(...)",__FILE__,__LINE__,
                        "single block does not fit in batch"); 
                    
            }

            //std::cout << "key: ";
            //for(size_t i = 0; i < it.key().size(); ++i)
            //{
                //std::cout << it.key()[i] << ",";
            //}
            //std::cout << "\n";
            size_t block_idx = it.key()[0];
            if(block_idx != prev_block_idx)
            {
                n_elem += this_block_idx_subtotal;
                this_block_idx_subtotal = 0;
                prev_block_idx = block_idx;
            }

            this_block_idx_subtotal += this_block_contrib;
            if(n_elem + this_block_idx_subtotal > max_n_elem)
            {
                //std::cout << "===============BATCH=================\n";
                batches.push_back(idx_pair(start_idx,block_idx));
                start_idx = block_idx;
                n_elem = 0;
            }
        }
    }
    //Handle last batch
    batches.push_back(idx_pair(start_idx,n_blocks));

    return batches;
}

template<size_t N>
size_t sparse_bispace<N>::get_batch_size(size_t subspace_idx,const idx_pair& batch) const
{
    bool is_sparse = false;
    size_t idx_grp = get_index_group_containing_subspace(subspace_idx);
    size_t target_sparse_grp;
    size_t idx_grp_offset = get_index_group_offset(idx_grp);
    std::vector<size_t>::const_iterator sparse_pos = find(m_sparse_indices_sets_offsets.begin(),m_sparse_indices_sets_offsets.end(),idx_grp_offset);
    if(sparse_pos != m_sparse_indices_sets_offsets.end())
    {
        is_sparse = true;
        target_sparse_grp = distance(m_sparse_indices_sets_offsets.begin(),sparse_pos);
    }

    size_t scale_fac = 1;
    for(size_t outer_idx_grp = 0; outer_idx_grp < idx_grp; ++outer_idx_grp)
    {
        scale_fac *= get_index_group_dim(outer_idx_grp);
    }
    for(size_t inner_idx_grp = idx_grp+1; inner_idx_grp < get_n_index_groups(); ++inner_idx_grp)
    {
        scale_fac *= get_index_group_dim(inner_idx_grp);
    }

    size_t batch_size = 0;
    if(!is_sparse)
    {
        for(size_t block_idx = batch.first; block_idx < batch.second; ++block_idx)
        {
            batch_size += m_subspaces[subspace_idx].get_block_size(block_idx)*scale_fac;
        }
    }
    else
    {
        size_t min = m_sparse_indices_sets_offsets[target_sparse_grp];
        size_t tree_subspace = subspace_idx - min;
        const sparse_block_tree& tree = m_sparse_block_trees[target_sparse_grp];
        for(sparse_block_tree::const_iterator it = tree.begin(); it != tree.end(); ++it)
        {
            size_t block_idx = it.key()[tree_subspace];
            if((batch.first <= block_idx) && (block_idx < batch.second))
            {
                batch_size += (*it)[0].second*scale_fac;
            }
        }
    }

    return batch_size;
}


template<size_t N> template<size_t L> 
sparse_bispace<N>::sparse_bispace(const sparse_bispace<N-L+1>& lhs, const sparse_bispace<L>& rhs)
{
    //Copy all subspaces,skipping the shared one
    for(size_t i = 0; i < N - L + 1; ++i)
    {
        m_subspaces.push_back(lhs.m_subspaces[i]);
    }
    for(size_t i = 1; i < L; ++i)
    {
        m_subspaces.push_back(rhs.m_subspaces[i]);
    }

    //We actually need to fuse sparse trees if there is overlap at the end of lhs and beginning of rhs
    bool fuse_sparsity = false;
    if(lhs.m_sparse_indices_sets_offsets.size() > 0)
    {
        size_t last_lhs_group_offset  = lhs.m_sparse_indices_sets_offsets.back();
        size_t last_lhs_group_order =  lhs.m_sparse_block_trees.back().get_order();
        size_t lhs_sparsity_end = last_lhs_group_offset + last_lhs_group_order;
        if((lhs_sparsity_end == (N-L+1)) && rhs.m_sparse_indices_sets_offsets[0] == 0)
        {
            fuse_sparsity = true;
        }
    }

    //Absorb all the sparsity, then patch it up later to account for fusion
    absorb_sparsity(lhs);
    absorb_sparsity(rhs,N-L);
    
    //Patch up the sparsity to account for fusion if appropriate
    if(fuse_sparsity)
    {
        size_t last_lhs_tree_idx = lhs.m_sparse_block_trees.size() - 1;
        size_t first_rhs_tree_idx = last_lhs_tree_idx + 1;
        m_sparse_block_trees[last_lhs_tree_idx] = m_sparse_block_trees[last_lhs_tree_idx].fuse(m_sparse_block_trees[first_rhs_tree_idx]);
        m_sparse_block_trees[last_lhs_tree_idx].set_offsets_sizes_nnz(m_subspaces);

        //delete the no longer needed rhs tree information
        m_sparse_indices_sets_offsets.erase(m_sparse_indices_sets_offsets.begin() + first_rhs_tree_idx);
        m_sparse_block_trees.erase(m_sparse_block_trees.begin() + first_rhs_tree_idx);
    }

    init();
}

template<size_t N> template<size_t L> 
sparse_bispace<N+L-1> sparse_bispace<N>::fuse(const sparse_bispace<L>& rhs) const
{
    if(m_subspaces[N-1] != rhs[0])
    {
        throw bad_parameter(g_ns,"sparse_bispace<N>","fuse(...)",
            __FILE__,__LINE__,"fuse point doesn't match"); 
    }
    return sparse_bispace<N+L-1>(*this,rhs);
}

template<size_t N>
const sparse_block_tree& sparse_bispace<N>::get_sparse_group_tree(size_t group_idx) const
{
    if(group_idx > (m_sparse_block_trees.size() - 1))
    {
        throw bad_parameter(g_ns,"sparse_bispace<N>","get_sparse_group_tree(...)",
            __FILE__,__LINE__,"group_idx too large"); 
    }
    return m_sparse_block_trees[group_idx];
}

template<size_t N>
size_t sparse_bispace<N>::get_sparse_group_offset(size_t group_idx) const
{
    if(group_idx > (m_sparse_block_trees.size() - 1))
    {
        throw bad_parameter(g_ns,"sparse_bispace<N>","get_sparse_group_tree(...)",
            __FILE__,__LINE__,"group_idx too large"); 
    }
    return m_sparse_indices_sets_offsets[group_idx];
}



template<size_t N>
size_t sparse_bispace<N>::get_index_group_order(size_t grp_idx) const
{
    if(grp_idx == m_index_group_offsets.size() - 1)
    {
        return N - m_index_group_offsets.back();
    }
    else
    {
        return m_index_group_offsets[grp_idx+1] - m_index_group_offsets[grp_idx];
    }
}

template<size_t N>
size_t sparse_bispace<N>::get_index_group_containing_subspace(size_t subspace_idx) const
{
    if(subspace_idx >= N)
    {
        throw bad_parameter(g_ns,"sparse_bispace<N>","get_index_group_containing_subspace()",
            __FILE__,__LINE__,"subspace idx too large"); 
    } 

    size_t index_group;
    for(size_t i = 0; i < m_index_group_offsets.size(); ++i)
    {
        if(m_index_group_offsets[i] <= subspace_idx)
        {
            if(i == m_index_group_offsets.size() - 1)
            {
                if(subspace_idx < N)
                {
                    index_group = i;
                    break;
                }
            }
            else if(m_index_group_offsets[i+1] > subspace_idx)
            {
                index_group = i;
                break;
            }
        }
    }
    return index_group;
}


template<size_t N>
bool sparse_bispace<N>::operator==(const sparse_bispace<N>& rhs) const
{
    //Check that subspaces are equivalent
    for(int i = 0; i < N; ++i)
    {
        if(m_subspaces[i] != rhs.m_subspaces[i])
        {
            return false;
        }
    }

    //Check that trees are equivalent
    if(m_sparse_indices_sets_offsets.size() != rhs.m_sparse_indices_sets_offsets.size())
    {
        return false;
    }
    for(size_t i = 0; i < m_sparse_indices_sets_offsets.size(); ++i)
    {
        if(m_sparse_indices_sets_offsets[i] != rhs.m_sparse_indices_sets_offsets[i])
        {
            return false;
        }
    }

    for(size_t i = 0; i < m_sparse_block_trees.size(); ++i)
    {
        if(m_sparse_block_trees[i] != rhs.m_sparse_block_trees[i])
        {
            return false;
        }
    }

    return true;
}

template<size_t N>
bool sparse_bispace<N>::operator!=(const sparse_bispace<N>& rhs) const
{
    return !(*this == rhs);
}

template<size_t N>
const char *sparse_bispace<N>::k_clazz = "sparse_bispace<N>";

//These two methods require the definition of sparse_bispace<2>, so they are stuck down here
inline sparse_bispace<1>::sparse_bispace(const sparse_bispace<2>& parent,size_t contract_idx)
{
    const sparse_bispace<1>& target = parent.m_subspaces[contract_idx ? 1 : 0];
    m_dim = target.get_dim();
    m_abs_indices.reserve(target.m_abs_indices.size());
    for(size_t i = 0;  i < target.m_abs_indices.size(); ++i)
    {
        m_abs_indices[i] = target.m_abs_indices[i];
    }
}

template<size_t M>
inline sparse_bispace<M+1> sparse_bispace<1>::operator|(const sparse_bispace<M>& rhs) const
{
    return sparse_bispace<M+1>(*this,rhs);
}

inline sparsity_expr<1,1> sparse_bispace<1>::operator%(const sparse_bispace<1>& rhs) const
{
    return sparsity_expr<1,1>(*this,rhs);
}

//TODO: Make this immutable, etc?? Need to make this class hard to abuse
//Type erasure class for sparse_bispaces.
//Used in functions that can take a number of sparse bispaces of different order as arguments
class sparse_bispace_any_order
{
private:
    class sparse_bispace_generic_i {
    public:
        virtual const sparse_bispace<1>& operator[](size_t idx) const = 0; 
        virtual size_t get_order() const = 0;
        virtual size_t get_nnz() const = 0; 
        virtual sparse_bispace_generic_i* clone() const = 0;
        virtual size_t get_n_sparse_groups() const  = 0;
        virtual const sparse_block_tree& get_sparse_group_tree(size_t group_idx) const  = 0;
        virtual size_t get_sparse_group_offset(size_t group_idx) const = 0; 
        virtual size_t get_n_index_groups() const = 0;
        virtual size_t get_index_group_offset(size_t grp_idx) const = 0;
        virtual size_t get_index_group_order(size_t grp_idx) const = 0;
        virtual size_t get_index_group_dim(size_t grp_idx) const = 0;
        virtual size_t get_index_group_containing_subspace(size_t subpsace_idx) const = 0;
        virtual void truncate_subspace(size_t subspace_idx,const idx_pair& bounds) = 0;
        virtual std::vector<idx_pair> get_batches(size_t subspace_idx,size_t max_n_elem) const = 0;
        virtual size_t get_batch_size(size_t subspace_idx,const idx_pair& batch) const = 0;

        virtual bool equals(const sparse_bispace_generic_i* rhs) const = 0;

        virtual ~sparse_bispace_generic_i() { };
    };

    template<size_t N>
    class sparse_bispace_generic_wrapper : public sparse_bispace_generic_i {
    private:
        sparse_bispace<N> m_bispace;
    public:
        sparse_bispace_generic_wrapper(const sparse_bispace<N>& bispace) : m_bispace(bispace) {};

        const sparse_bispace<1>& operator[](size_t idx) const { return m_bispace[idx]; }
        size_t get_order() const { return N; }
        size_t get_nnz() const { return m_bispace.get_nnz(); }
        size_t get_n_sparse_groups() const  { return m_bispace.get_n_sparse_groups(); }
        const sparse_block_tree& get_sparse_group_tree(size_t group_idx) const { return m_bispace.get_sparse_group_tree(group_idx); };
        size_t get_sparse_group_offset(size_t group_idx) const { return m_bispace.get_sparse_group_offset(group_idx); }
        size_t get_n_index_groups() const { return m_bispace.get_n_index_groups(); }
        size_t get_index_group_offset(size_t grp_idx) const { return m_bispace.get_index_group_offset(grp_idx); }
        size_t get_index_group_order(size_t grp_idx) const { return m_bispace.get_index_group_order(grp_idx); }
        size_t get_index_group_dim(size_t grp_idx) const { return m_bispace.get_index_group_dim(grp_idx); }
        size_t get_index_group_containing_subspace(size_t subspace_idx) const { return m_bispace.get_index_group_containing_subspace(subspace_idx); }
        void truncate_subspace(size_t subspace_idx,const idx_pair& bounds) { return m_bispace.truncate_subspace(subspace_idx,bounds); }
        virtual std::vector<idx_pair> get_batches(size_t subspace_idx,size_t max_n_elem) const { return m_bispace.get_batches(subspace_idx,max_n_elem); }
        virtual size_t get_batch_size(size_t subspace_idx,const idx_pair& batch) const { return m_bispace.get_batch_size(subspace_idx,batch); }

        //Same order is assured upstream
        bool equals(const sparse_bispace_generic_i* rhs) const { return m_bispace == static_cast< const sparse_bispace_generic_wrapper<N>* >(rhs)->m_bispace; }


        sparse_bispace_generic_i* clone() const { return new sparse_bispace_generic_wrapper(m_bispace); }
    };

    sparse_bispace_generic_i* m_spb_ptr;
    
    //This needs to exist given the current implementation of sequence<>, so we hack and hide it
    sparse_bispace_any_order() { m_spb_ptr = NULL; };
public:
    
    //Constructor
    template<size_t N>
    sparse_bispace_any_order(const sparse_bispace<N>& bispace) { m_spb_ptr = new sparse_bispace_generic_wrapper<N>(bispace); }

    //Copy constructor
    sparse_bispace_any_order(const sparse_bispace_any_order& rhs) { rhs.m_spb_ptr ? m_spb_ptr = rhs.m_spb_ptr->clone() : m_spb_ptr = 0; }

    //Overloaded assignment operator
    sparse_bispace_any_order& operator=(const sparse_bispace_any_order& rhs) { if(m_spb_ptr) { delete m_spb_ptr; } rhs.m_spb_ptr ? m_spb_ptr = rhs.m_spb_ptr->clone() : m_spb_ptr = 0; return *this; }

    const sparse_bispace<1>& operator[](size_t idx) const { return (*m_spb_ptr)[idx]; }
    size_t get_order() const { return m_spb_ptr->get_order(); }
    size_t get_nnz() const { return m_spb_ptr->get_nnz(); }
    size_t get_n_sparse_groups() const { return m_spb_ptr->get_n_sparse_groups(); }
    const sparse_block_tree& get_sparse_group_tree(size_t group_idx) const { return m_spb_ptr->get_sparse_group_tree(group_idx); }
    size_t get_sparse_group_offset(size_t group_idx) const { return m_spb_ptr->get_sparse_group_offset(group_idx); } 
    size_t get_n_index_groups() const { return m_spb_ptr->get_n_index_groups(); }
    size_t get_index_group_offset(size_t grp_idx) const { return m_spb_ptr->get_index_group_offset(grp_idx); }
    size_t get_index_group_order(size_t grp_idx) const { return m_spb_ptr->get_index_group_order(grp_idx); }
    size_t get_index_group_dim(size_t grp_idx) const { return m_spb_ptr->get_index_group_dim(grp_idx); }
    size_t get_index_group_containing_subspace(size_t subspace_idx) const { return m_spb_ptr->get_index_group_containing_subspace(subspace_idx); }
    
    void truncate_subspace(size_t subspace_idx,const idx_pair& bounds) { return m_spb_ptr->truncate_subspace(subspace_idx,bounds); }
    std::vector<idx_pair> get_batches(size_t subspace_idx,size_t max_n_elem) const { return m_spb_ptr->get_batches(subspace_idx,max_n_elem); }
    size_t get_batch_size(size_t subspace_idx,const idx_pair& batch) const { return m_spb_ptr->get_batch_size(subspace_idx,batch); }


    //We have to check NULL bcs of stupid default constructor hack
    virtual ~sparse_bispace_any_order() { if(m_spb_ptr != NULL) { delete m_spb_ptr; } };

    bool operator==(const sparse_bispace_any_order& rhs) const { if(get_order() != rhs.get_order()) { return false; } return m_spb_ptr->equals(rhs.m_spb_ptr); }
    bool operator!=(const sparse_bispace_any_order& rhs) const { return !(*this == rhs); }

    //For default constructor hack
    template<size_t N,typename T>
    friend class sequence;
};

} // namespace libtensor
#endif

#endif // SPARSE_BISPACE_H
