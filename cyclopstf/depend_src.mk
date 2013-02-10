
SHARED_OBJ_DIR = ../../objs_dist_tensor/

CTR_COMM_DIR = ../ctr_comm
CTR_SEQ_DIR = ../ctr_seq
DIST_TENSOR_DIR = ../dist_tensor

CTR_COMM_SRC = $(CTR_COMM_DIR)/ctr_simple.cxx \
               $(CTR_COMM_DIR)/ctr_1d_sqr_bcast.cxx \
               $(CTR_COMM_DIR)/ctr_2d_sqr_bcast.cxx \
               $(CTR_COMM_DIR)/ctr_2d_rect_bcast.cxx \
               $(CTR_COMM_DIR)/ctr_2d_general.cxx \
               $(CTR_COMM_DIR)/strp_tsr.cxx \
               $(CTR_COMM_DIR)/ctr_tsr.cxx \
               $(CTR_COMM_DIR)/sum_tsr.cxx \
               $(CTR_COMM_DIR)/scale_tsr.cxx

DIST_TENSOR_SRC = $(DIST_TENSOR_DIR)/cyclopstf.cxx \
                  $(DIST_TENSOR_DIR)/cyclopstf.hpp \
                  $(DIST_TENSOR_DIR)/mach.h \
                  $(DIST_TENSOR_DIR)/scala_backend.cxx \
                  $(DIST_TENSOR_DIR)/dist_tensor_internal.cxx \
                  $(DIST_TENSOR_DIR)/dist_tensor_internal.h \
                  $(DIST_TENSOR_DIR)/dt_aux_topo.hxx \
                  $(DIST_TENSOR_DIR)/dt_aux_permute.hxx \
                  $(DIST_TENSOR_DIR)/dt_aux_map.hxx \
                  $(DIST_TENSOR_DIR)/dt_aux_rw.hxx \
                  $(DIST_TENSOR_DIR)/dt_aux_sort.hxx \
                  $(DIST_TENSOR_DIR)/dist_tensor_inner.cxx \
                  $(DIST_TENSOR_DIR)/dist_tensor_fold.cxx \
                  $(DIST_TENSOR_DIR)/dist_tensor_map.cxx \
                  $(DIST_TENSOR_DIR)/dist_tensor_op.cxx 
                   
CTR_SEQ_SRC = $(CTR_SEQ_DIR)/sym_seq_ctr_ref.hxx \
              $(CTR_SEQ_DIR)/sym_seq_sum_ref.hxx \
              $(CTR_SEQ_DIR)/sym_seq_scl_ref.hxx \
              $(CTR_SEQ_DIR)/sym_seq_ctr_inner.hxx \
              $(CTR_SEQ_DIR)/sym_seq_sum_inner.hxx 
