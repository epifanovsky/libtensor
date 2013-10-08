#ifndef BLOCK_PRINTER_H
#define BLOCK_PRINTER_H

#include "block_kernel_i.h"
#include <sstream>

namespace libtensor {

template<typename T = double>
class block_printer : public block_kernel_i<0,1,T> {
private:
    std::stringstream m_ss;
    //Used to recursively traverse block to print individual elements
    void _process_dimension(const dim_list& dims,T* data_ptr,size_t offset = 0,size_t dim_idx = 0);
public:

    //Get the string representation of the last block processed
    std::string str() const { return m_ss.str(); };

    //Default constructor
    block_printer() { };
    //Copy constructor
    block_printer(const block_printer<T>& rhs) { m_ss.str(rhs.str()); };

    //Returns a pointer to a copy of this object
    block_kernel_i<0,1,T>* clone() const { return (block_kernel_i<0,1,T>*) new block_printer(*this); };  

    //Abstract base class methods
    void operator()(sequence<0, T*>& output_ptrs, 
                    sequence<1, T*>& input_ptrs,
                    sequence<0, dim_list>& output_dims,
                    sequence<1, dim_list>& input_dims);
};

template<typename T> 
void block_printer<T>::_process_dimension(const dim_list& dims,T* data_ptr,size_t offset,size_t dim_idx)
{
    //Base case
    if(dim_idx == (dims.size() - 1))
    {
        T* inter_data_ptr = data_ptr + offset;  
        for(int i = 0; i < dims.back(); ++i)
        {
            m_ss << ' ' << *inter_data_ptr;
            inter_data_ptr += 1;
        }
        m_ss << std::endl;
        return;
    }
    else
    {
        //Delimit blocks of varying dimensions by the corresponding number of newlines
        //But must skip first block for aesthetics 
        if(offset != 0)
        {
            m_ss << std::endl;
        }
        size_t inner_size = 1;
        for(int i = dim_idx+1; i < dims.size(); ++i)
        {
            inner_size *= dims[i];
        }
        for(int i = 0; i < dims[dim_idx]; ++i)
        {
            _process_dimension(dims,data_ptr,offset,dim_idx+1);
            offset += inner_size;
        }
    }
}

template<typename T>
void block_printer<T>::operator()(sequence<0, T*>& output_ptrs, 
                                  sequence<1, T*>& input_ptrs,
                                  sequence<0, dim_list>& output_dims,
                                  sequence<1, dim_list >& input_dims)
{
    _process_dimension(input_dims[0],input_ptrs[0]);
}

} // namespace libtensor

#endif /* BLOCK_PRINTER_H */
