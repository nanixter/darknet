#ifndef LIVESTREAMDETECTOR_QUEUE_H
#define LIVESTREAMDETECTOR_QUEUE_H
#include <queue>
#include <mutex>
#include <condition_variable>

namespace LiveStreamDetector {

	template <class T>
	class MutexQueue {
	public:
		void Init() {
			this->done.store(false, std::memory_order_release);
		}

		void push_back(T &elem) {
			std::lock_guard<std::mutex> lock(this->mutex);
			this->queue.push(elem);
			this->cv.notify_one();
		}

		void push_back(std::vector<T> &elems) {
			std::lock_guard<std::mutex> lock(this->mutex);
			for (auto elemIterator = elems.begin(); elemIterator != elems.end(); elemIterator++) {
				this->queue.push(*elemIterator);
			}
			this->cv.notify_one();
		}

		void notify_all() {
			this->done.store(true, std::memory_order_release);
			this->cv.notify_all();
		}

		// pops 1 element
		void pop_front(T &elem) {
			std::unique_lock<std::mutex> lock(this->mutex);
			if(this->queue.empty())
				cv.wait(lock, [this](){ return (!this->queue.empty())||this->done.load(std::memory_order_acquire); });

			// Once the cv wakes us up....
			if(!this->queue.empty()) {
				elem = (this->queue.front());
				this->queue.pop();
			}
		}

		// Pops upto N elements
		void pop_front(std::vector<T> &elems, int &numElems) {
			std::unique_lock<std::mutex> lock(this->mutex);
			if(this->queue.empty())
				cv.wait(lock, [this](){ return (!this->queue.empty())||this->done.load(std::memory_order_acquire); });

			// Once the cv wakes us up....
			int numPopped = 0;
			while( !this->queue.empty() && (numPopped < numElems) ) {
				elems.insert(elems.end(), this->queue.front());
				this->queue.pop();
				numPopped++;
			}
			numElems = numPopped;
		}

	private:
		std::atomic_bool done;
		std::queue<T> queue;
		std::mutex mutex;
		std::condition_variable cv;

	}; // class DetectionQueue

} // namespace

#endif // QUEUE_H
