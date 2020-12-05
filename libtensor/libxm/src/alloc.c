/*
 * Copyright (c) 2014-2018 Ilya Kaliman
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef XM_USE_MPI
#include <mpi.h>
#endif

#include "alloc.h"
#include "bitmap.h"
#include "util.h"

/* Data is allocated in 512 KiB chunks. */
#define XM_PAGE_SIZE (512ULL * 1024)

/* Pagefile growth when no more space is available. */
#define XM_GROW_SIZE (256ULL * 1024 * 1024 * 1024)

struct xm_allocator {
	int fd;
	int mpirank;
	char *path;
	size_t file_bytes;
	unsigned char *pages;
#ifdef _OPENMP
	omp_lock_t mutex;
#endif
};

/* 64-bit data_ptr handle: 32-bit size in number of pages + 32-bit file offset
 * in page size. */
static uint64_t
make_data_ptr(uint64_t offset, uint64_t npages)
{
	return (offset | (npages << 32));
}

/* Return block offset in bytes. */
static size_t
get_block_offset(uint64_t data_ptr)
{
	return (data_ptr & ((1ULL << 32) - 1)) * XM_PAGE_SIZE;
}

static size_t
get_block_npages(uint64_t data_ptr)
{
	return (data_ptr >> 32);
}

static int
extend_file(xm_allocator_t *allocator)
{
	size_t oldsize, newsize;

	oldsize = allocator->file_bytes / XM_PAGE_SIZE / 8;
	if (allocator->file_bytes % (XM_PAGE_SIZE * 8))
		oldsize++;
	allocator->file_bytes = allocator->file_bytes > XM_GROW_SIZE ?
	    allocator->file_bytes + XM_GROW_SIZE :
	    allocator->file_bytes * 2;
	newsize = allocator->file_bytes / XM_PAGE_SIZE / 8;
	if (allocator->file_bytes % (XM_PAGE_SIZE * 8))
		newsize++;

	if (ftruncate(allocator->fd, (off_t)allocator->file_bytes)) {
		perror("ftruncate");
		return (1);
	}
	if ((allocator->pages = realloc(allocator->pages, newsize)) == NULL) {
		perror("realloc");
		return (1);
	}
	memset(allocator->pages + oldsize, 0, newsize - oldsize);
	return (0);
}

static uint64_t
find_pages(xm_allocator_t *allocator, size_t n_pages)
{
	size_t i, n_free, n_total, start;

	assert(n_pages > 0);

	n_total = allocator->file_bytes / XM_PAGE_SIZE;
	for (start = 0; start < n_total; start++)
		if (!bitmap_test(allocator->pages, start))
			break;
	if (start == n_total)
		return (XM_NULL_PTR);
	for (i = start, n_free = 0; i < n_total; i++) {
		n_free = bitmap_test(allocator->pages, i) ? 0 : n_free + 1;
		if (n_free == n_pages) {
			uint64_t offset;
			for (offset = i + 1 - n_free; offset <= i; offset++)
				bitmap_set(allocator->pages, offset);
			offset = (uint64_t)(i + 1 - n_free);
			return make_data_ptr(offset, n_pages);
		}
	}
	return (XM_NULL_PTR);
}

static uint64_t
allocate_pages(xm_allocator_t *allocator, size_t size_bytes)
{
	size_t n_pages;
	uint64_t ptr;

	if (size_bytes == 0)
		return (XM_NULL_PTR);

	n_pages = (size_bytes + XM_PAGE_SIZE - 1) / XM_PAGE_SIZE;

	while ((ptr = find_pages(allocator, n_pages)) == XM_NULL_PTR)
		if (extend_file(allocator))
			return (XM_NULL_PTR);
	return (ptr);
}

xm_allocator_t *
xm_allocator_create(const char *path)
{
	xm_allocator_t *allocator;

#ifdef XM_USE_MPI
	if (path == NULL)
		fatal("data must be on a shared filesystem when using MPI");
#endif
	if ((allocator = calloc(1, sizeof(*allocator))) == NULL) {
		perror("calloc");
		return (NULL);
	}
#ifdef XM_USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &allocator->mpirank);
#endif
	if (path) {
		allocator->file_bytes = XM_PAGE_SIZE;
		if ((allocator->pages = calloc(1, 1)) == NULL)
			fatal("out of memory");
		if (allocator->mpirank == 0) {
			if ((allocator->fd = open(path, O_CREAT|O_RDWR,
			    S_IRUSR|S_IWUSR)) == -1) {
				perror("open");
				free(allocator);
				return (NULL);
			}
			if (ftruncate(allocator->fd,
			    (off_t)allocator->file_bytes)) {
				perror("ftruncate");
				if (close(allocator->fd))
					perror("close");
				free(allocator);
				return (NULL);
			}
#ifdef XM_USE_MPI
			MPI_Barrier(MPI_COMM_WORLD);
#endif
		} else {
#ifdef XM_USE_MPI
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			if ((allocator->fd = open(path, O_RDWR)) == -1) {
				perror("open");
				free(allocator);
				return (NULL);
			}
		}
		if ((allocator->path = strdup(path)) == NULL) {
			perror("strdup");
			if (close(allocator->fd))
				perror("close");
			free(allocator);
			return (NULL);
		}
	}
#ifdef _OPENMP
	omp_init_lock(&allocator->mutex);
#endif
	return (allocator);
}

const char *
xm_allocator_get_path(xm_allocator_t *allocator)
{
	return (allocator->path);
}

uint64_t
xm_allocator_allocate(xm_allocator_t *allocator, size_t size_bytes)
{
	uint64_t data_ptr = XM_NULL_PTR;
	void *data;

	if (allocator->mpirank != 0) {
#ifdef XM_USE_MPI
		MPI_Bcast(&data_ptr, 1, MPI_UNSIGNED_LONG_LONG, 0,
		    MPI_COMM_WORLD);
#endif
		return (data_ptr);
	}
#ifdef _OPENMP
	omp_set_lock(&allocator->mutex);
#endif
	if (allocator->path) {
		data_ptr = allocate_pages(allocator, size_bytes);
	} else {
		if ((data = malloc(size_bytes)) == NULL) {
			perror("malloc");
			data_ptr = XM_NULL_PTR;
		} else
			data_ptr = (uint64_t)data;
	}
#ifdef _OPENMP
	omp_unset_lock(&allocator->mutex);
#endif
#ifdef XM_USE_MPI
	MPI_Bcast(&data_ptr, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
#endif
	return (data_ptr);
}

/* Maximum size for single pread/pwrite. */
#define MAXSIZE (1<<30)

void
xm_allocator_read(xm_allocator_t *allocator, uint64_t data_ptr,
    void *mem, size_t size_bytes)
{
	ssize_t read_bytes;
	off_t offset;

	if (data_ptr == XM_NULL_PTR)
		fatal("data pointer is NULL");
	if (allocator->path == NULL) {
		memcpy(mem, (const void *)data_ptr, size_bytes);
		return;
	}
	offset = (off_t)get_block_offset(data_ptr);
	while (size_bytes > 0) {
		size_t size = size_bytes > MAXSIZE ? MAXSIZE : size_bytes;
		read_bytes = pread(allocator->fd, mem, size, offset);
		if (read_bytes != (ssize_t)size)
			fatal("pread");
		mem = (char *)mem + size;
		offset += size;
		size_bytes -= size;
	}
}

void
xm_allocator_write(xm_allocator_t *allocator, uint64_t data_ptr,
    const void *mem, size_t size_bytes)
{
	ssize_t write_bytes;
	off_t offset;

	if (data_ptr == XM_NULL_PTR)
		fatal("data pointer is NULL");
	if (allocator->path == NULL) {
		memcpy((void *)data_ptr, mem, size_bytes);
		return;
	}
	offset = (off_t)get_block_offset(data_ptr);
	while (size_bytes > 0) {
		size_t size = size_bytes > MAXSIZE ? MAXSIZE : size_bytes;
		write_bytes = pwrite(allocator->fd, mem, size, offset);
		if (write_bytes != (ssize_t)size)
			fatal("pwrite");
		mem = (const char *)mem + size;
		offset += size;
		size_bytes -= size;
	}
}

void
xm_allocator_deallocate(xm_allocator_t *allocator, uint64_t data_ptr)
{
	if (allocator->mpirank != 0)
		return;
	if (data_ptr == XM_NULL_PTR)
		return;
#ifdef _OPENMP
	omp_set_lock(&allocator->mutex);
#endif
	if (allocator->path) {
		size_t i, start;
		size_t offset = get_block_offset(data_ptr);
		size_t npages = get_block_npages(data_ptr);
		assert(offset % XM_PAGE_SIZE == 0);
		start = offset / XM_PAGE_SIZE;
		for (i = 0; i < npages; i++)
			bitmap_clear(allocator->pages, start + i);
	} else {
		free((void *)data_ptr);
	}
#ifdef _OPENMP
	omp_unset_lock(&allocator->mutex);
#endif
}

void
xm_allocator_destroy(xm_allocator_t *allocator)
{
	if (allocator == NULL)
		return;
	if (allocator->mpirank == 0) {
		if (allocator->path) {
			if (close(allocator->fd))
				perror("close");
			if (unlink(allocator->path))
				perror("unlink");
		}
	}
#ifdef _OPENMP
	omp_destroy_lock(&allocator->mutex);
#endif
	free(allocator->path);
	free(allocator->pages);
	free(allocator);
}
