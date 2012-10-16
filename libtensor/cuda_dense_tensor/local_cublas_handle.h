#include <cublas_v2.h>
#include <libutil/threads/tls.h>

namespace libtensor {


/** \brief Stores a thread-local cuBLAS handle

    Each worker thread owns its own CUDA stream and a corresponding cuBLAS
    handle. This structure returns the handle that belongs to the current
    thread.

    \ingroup libtensor_cuda_tod
 **/
class local_cublas_handle {
private:
    cublasHandle_t m_handle; //!< cuBLAS handle

public:
    /** \brief Initializes the handle
     **/
    local_cublas_handle();

    /** \brief Frees the handle
     **/
    ~local_cublas_handle();

public:
    /** \brief Returns the cuBLAS handle specific to current thread
     **/
    static const cublasHandle_t &get() {
        return libutil::tls<local_cublas_handle>::get_instance().get().m_handle;
    }

};


} // namespace libtensor

