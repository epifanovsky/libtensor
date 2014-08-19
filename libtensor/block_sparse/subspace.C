#include "subspace.h"

namespace libtensor {

const char* subspace::k_clazz = "subspace";

size_t subspace::get_dim() const
{
    return m_dim;
}

size_t subspace::get_n_blocks() const
{
    return m_abs_indices.size();
}

void subspace::split(const std::vector<size_t>& split_points) throw(out_of_bounds)
{
    if(split_points.size() < 1 || split_points.size() > (m_dim - 1))
    {
        throw out_of_bounds(g_ns,k_clazz,"split(...)",
                __FILE__,__LINE__,"Must have 1 <= # of split points <= dim - 1"); 
    }

    for(int i = 0; i < split_points.size(); ++i)
    {
        size_t split_point = split_points[i];
        if(split_point > (m_dim - 1))
        {
            throw out_of_bounds(g_ns,k_clazz,"split(...)",
                    __FILE__,__LINE__,"Split point indices cannot exceed (dim - 1)"); 
        }
        else if(split_point <= m_abs_indices.back())
        {
            throw out_of_bounds(g_ns,k_clazz,"split(...)",
                    __FILE__,__LINE__,"Split point indices must be strictly increasing"); 
        }
        m_abs_indices.push_back(split_point);
    }
}

size_t subspace::get_block_size(size_t block_idx) const throw(out_of_bounds)
{
    if(block_idx > (m_abs_indices.size() - 1))
    {
        throw out_of_bounds(g_ns,k_clazz,"get_block_size(size_t block_idx)",
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


size_t subspace::get_block_abs_index(size_t block_idx) const throw(out_of_bounds)
{
    if(block_idx > (m_abs_indices.size() - 1))
    {
        throw out_of_bounds(g_ns,k_clazz,"get_block_abs_index(size_t block_idx)",
                __FILE__,__LINE__,"Cannot pass block_idx greater than (# of blocks - 1)"); 
    }
    return m_abs_indices[block_idx];
}

} // namespace libtensor
