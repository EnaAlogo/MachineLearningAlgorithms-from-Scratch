#include <unordered_map>
#include <vector>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <utility>
#include <algorithm>

typedef int32_t i32;
typedef uint8_t u8;
typedef int16_t i16;
typedef uint16_t u16;

#define UP x-1,y
#define DOWN x+1,y
#define LEFT x , y-1
#define RIGHT  x , y+1

#define ISVALID(pos) canmove(matrix , pos ,set)

#define MEM_CLR() \
               for (auto& node : set)\
               delete node.second;\
               while (!q.empty())\
               {\
               	Graph* garbage = q.front();\
               	delete garbage;\
               	q.pop();\
               }\

namespace AI_ERGASIA_UTILS {
  


	template <typename b>
	struct STRvGraph {

		STRvGraph* trail;
		b x;
		b y;

		STRvGraph(STRvGraph* pointer, b x, b y)
			:trail(pointer), x(x), y(y) {};

		~STRvGraph() = default;

		/*κακό data moving κανονικά θα έπρεπε να περνάω rvalue αλλα το front της ουράς
		επέστρεφε για κάποιο λόγο const ref ακόμα και με το std move cast*/
		void operator = (const STRvGraph& other)
		{
			x = other.x;
			y = other.y;
			trail = other.trail;
			other.trail = nullptr;
		}

		//βρίσκει το μοναδικό path κάποιου κόμβου από το origin
		static void RETRACT_TO_ORIGIN(STRvGraph* node, std::vector<std::pair<b, b>>& path);


	};

	template <typename b>
	void STRvGraph<b> ::RETRACT_TO_ORIGIN(STRvGraph* node, std::vector<std::pair<b, b>>& path)
	{
		STRvGraph* Node = node;
		while (Node)
		{
			path.push_back({ Node->x,Node->y });
			Node = Node->trail;
		}
		std::reverse(path.begin(), path.end());
	}

	template <class a, class b>
	inline bool constexpr const canmove(const std::vector<std::vector<a>>& matrix, const std::pair<b, b> pos, std::unordered_map<a const*, STRvGraph<b>*>& set)
	{
		return (pos.first < matrix.size() && pos.second < matrix[0].size() && matrix[pos.first][pos.second]
			&& set.find(&matrix[pos.first][pos.second]) == set.end());
	}
	template <class a, class b>
	inline bool constexpr const canmove(const std::vector<std::vector<a>>& matrix, const std::pair<b, b> pos, std::unordered_set<a const* >& set)
	{
		return (pos.first < matrix.size() && pos.second < matrix[0].size() && matrix[pos.first][pos.second]
			&& set.find(&matrix[pos.first][pos.second]) == set.end());
	}

	template<class a>
	const std::vector<std::vector<std::vector<a>>> make_testcases()
	{
		return {
		 {
			 {1,1,0,0,1,1,1},
			 {1,1,1,0,'G',1,1},
			 {1,0,0,0,0,1,0},
			 {1,1,1,1,0,1,1},
			 {1,1,0,1,1,1,1}
		 },


		 {
			 {1,1,0,0,1,1,1},
			 {1,1,1,1,'G',1,1},
			 {1,0,0,0,0,1,0},
			 {1,1,1,1,0,1,1},
			 {1,1,0,1,1,1,1}
		 },


		 {
			 {1,1,0,0,1,1,1},
			 {1,1,0,1,'G',1,1},
			 {1,0,1,0,0,0,1},
			 {1,1,1,1,0,1,1},
			 {1,1,0,1,1,1,1}
		 },


		 {
			 {1,1,1,1,1,1,1},
			 {0,1,0,1,'G',1,1},
			 {1,1,1,0,0,0,1},
			 {1,1,1,1,0,1,1},
			 {1,1,0,1,1,1,1}
		 },


		 {
			 {1,1,1,1,1,1,1},
			 {0,1,0,1,'G',1,1},
			 {1,1,1,0,0,0,0},
			 {1,1,1,1,0,1,1},
			 {1,1,0,1,1,1,1}
		 },
		};

	}

	static constexpr inline const char* const ComingFrom(i32 x_coord, i32 y_coord, i32 x_prev, i32 y_prev)
	{
		if (x_coord + 1 == x_prev)
			return "up";
		else if (x_coord - 1 == x_prev)
			return "down";
		else if (y_coord + 1 == y_prev)
			return "left";
		else  return "right";
	}

	
	static std::ostream& operator << (std::ostream& os, std::pair< i32,  i32> printable)
	{
		const auto& [x, y] = printable;
		os << '(' << ' ' << x << ' ' << ',' << ' ' << y << ' ' << ')';
		return os;
	}


    static void printPath(const std::vector<std::vector<u8>>& matrix ,const std::vector<std::tuple<i32,i32,char const* const>>& path)
	{
		for (int i = 0; i < path.size() - 1; i++)
		{
			const auto& [x, y, _] = path[i];
			const char* const movingTo = std::get<2>(path[i + 1]);
			std::cout << std::pair{ x,y } << " moving to-> " << movingTo << '\n';
		}
		const auto& [x, y, _] = path.back();
		std::cout << std::pair{ x,y } << (" goal reached")<<'\n';
	}

	static const std::vector<std::tuple<i32, i32,char const* const>> AddPathDirections(const std::vector<std::pair<i16, i16>>& path)
	{
		std::vector<std::tuple< i32, i32,char const* const>> newPath;
		newPath.push_back({ path[0].first, path[0].second, "initial" });

		for (i32 i = 1; i < path.size(); i++)
		{
			auto& [x, y] = path[i];
			auto& [x_prev, y_prev] = path[i - 1];
			newPath.push_back({ x,y, ComingFrom(x, y, x_prev, y_prev) });
		}

		return newPath;
	}

	static void SHOW_PATH(const std::vector<std::vector<u8>>&matrix, const std::vector<std::pair<i16, i16>>& path)
	{
		const auto printablePath = AddPathDirections(path);
		printPath(matrix,printablePath);
		std::cout << "total steps : " << path.size() << '\n';
	}
}