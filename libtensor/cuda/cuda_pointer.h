#ifndef LIBTENSOR_CUDA_POINTER_H
#define LIBTENSOR_CUDA_POINTER_H


namespace libtensor {


/** \brief Simple allocator for CUDA GPU memory
    \tparam T Data type.

    This class allocates memory on nVidia CUDA GPUs using cudaMalloc/cudaFree.

    \ingroup libtensor_cuda
 **/
template<typename T>
struct cuda_pointer {

//		template <class P>
//		struct RemoveConst
//		{
//			typedef P type;
//		};
//
//		template <class P>
//		struct RemoveConst<const P>
//		{
//			typedef P type;
//		};

        T *p;
        cuda_pointer(T *p_ = 0) : p(p_) { }
        cuda_pointer(const cuda_pointer &p_) : p(p_.get_physical_pointer()) { }
//        cuda_pointer(const T *p_ = 0) : p(p_) { }
        bool operator==(cuda_pointer &p_) const { return p == p_.get_physical_pointer(); }

        bool operator!=(cuda_pointer &p_) const { return !(*this == p_); }


        T* get_physical_pointer() const {
        	return p;
        }
    } ; //!< Wrapped CUDA pointer type

//*
template<typename T>
struct cuda_pointer<const T> {

//		template <class P>
//		struct RemoveConst
//		{
//			typedef P type;
//		};
//
//		template <class P>
//		struct RemoveConst<const P>
//		{
//			typedef P type;
//		};

        const T *p;
        cuda_pointer(T *p_ = 0) : p(p_) { }
        cuda_pointer(const cuda_pointer<T> &p_) : p(p_.get_physical_pointer()) { }
//        cuda_pointer(const T *p_ = 0) : p(p_) { }
        bool operator==(cuda_pointer &p_) const { return p == p_.get_physical_pointer(); }

        bool operator!=(cuda_pointer &p_) const { return !(*this == p_); }

//        bool operator=(const cuda_pointer<T> &p_)  { return !(*this == p_); }


        const T* get_physical_pointer() {
        	return p;
        }
    } ; //!< Wrapped CUDA pointer type
//*/


} // namespace libtensor

#endif // LIBTENSOR_CUDA_POINTER_H
