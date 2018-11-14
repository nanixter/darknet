
#ifndef POINTERMAP_HPP
#define POINTERMAP_HPP

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

namespace LiveStreamDetector {
	template <class T>
	class PointerMap {
	public:

		void insert(T *elem, std::uint64_t elemNum)
		{
			std::lock_guard<std::mutex> lock(this->mutex);
			elements.emplace(std::make_pair(elemNum, elem));
			cv.notify_all();
		}

		bool getElem(T **elem, std::uint64_t elemNum)
		{
			std::unique_lock<std::mutex> lock(this->mutex);
			if (elements.empty())
				cv.wait(lock, [this](){ return !this->elements.empty(); });

			auto iterator = elements.find(elemNum);
			if (iterator == elements.end()) {
				return false;
			} else {
				*elem = iterator->second;
				return true;
			}
		}

		void remove(std::uint64_t frameNum)
		{
			std::lock_guard<std::mutex> lock(this->mutex);
			auto iterator = elements.find(frameNum);
			if (iterator != elements.end())
				delete iterator->second;
			elements.erase(frameNum);
		}

		int size() {
			std::lock_guard<std::mutex> lock(this->mutex);
			return elements.size();
		}

	private:
		std::unordered_map<std::uint64_t, T *> elements;
		std::mutex mutex;
		std::condition_variable cv;
	};
} // namespace
#endif //POINTERMAP_HPP