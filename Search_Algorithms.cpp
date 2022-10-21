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

std::vector<std::pair<i16,i16>> bfs(const std::vector<std::vector<u8>>& matrix, std::pair<i16, i16> position)
{
	/*η δομή STRvGraph που έφτιαξα είναι μια δομή σαν ανάποδο δέντρο στην οποια κάθε κόμβος πέρα από τα δεδομένα του
	x,y συντεταγμένες, συνδέεται και με τον προηγούμενο του ώστε όταν βρεθεί ο κόμβος στόχος
	να υπάρχει μοναδικό μονοπάτι από τον given κόμβο προς το origin-root
	*/
	using Graph = AI_ERGASIA_UTILS::STRvGraph<i16>;
	
	
	/*hashmap με κλειδιά τους pointers των κελιών του πινακα ώστε να μπορεί να ελέγχει
	εάν μια given θέση βρίκσεται στο κλειστό σύνολο σε Ο(1) μέση πολυπλοκότητα χρόνου,
	επίσης σώζει και όλους τους pointers των κόμβων που θα δημιουργηθούν ώστε να μπορούν 
	στο τέλος να διαγραφουν διότι η φύση της δομής σαν ένα ανάποδο δέντρο δεν δίνει πρόσβαση σε όλους τους 
	κόμβους*/
	std::unordered_map<u8 const* , Graph*> set;

	std::queue< Graph* > q;//η ουρα για την αναζήτηση

	//βάζουμε στην ουρα την αρχική κατάσταση
	q.push( new Graph{ nullptr ,position.first,position.second } );

	do
	{
		//βγάζουμε το πρώτο στοιχειο από την ουρα
		Graph* Node = std::move( q.front() );
		q.pop();//το pop σβήνει το στοιχειο και το front επιστρέφει αναφορά οποτε το μετακινώ για να μην σβηστούν τα δεδομένα
		const auto& [_, x, y] = *Node;
		position={ x,y };

		/*εάν δεν μπορούμε να μετακινηθούμε - δηλαδή εάν είτε δεν είμαστε μέσα στα όρια του πινακα
		είτε η θέση είναι εμπόδιο είτε η θέση βρίσκετε στο κλειστό σύνολο - 
		σε αυτή τη θέση φεύγουμε και την σβήνουμε απτην μνήμη*/
		if (!ISVALID(position))
		{
			delete Node;
			continue;
		}
        //διαφορετικά βάζουμε την θέση στο κλειστό σύνολο
		set.insert({ &matrix[x][y], Node });

		//εάν είναι ο στόχος σταματάμε και σώζουμε το path
		if (matrix[x][y] == 'G')
		{
			std::vector<std::pair<i16, i16>> path;

			/*εδώ βρίσκω το μονοπάτι απτoν given κόμβο(στόχο) στο origin 
			του δέντρου(δηλαδή την αρχική κατάσταση) και το αποθηκεύω*/
			Graph::RETRACT_TO_ORIGIN( Node , path );

	        //καθαρίζουμε την μνήμη και επιστρέφουμε το μονοπάτι
			MEM_CLR();
			
			return path;
		}

		//βάζουμε στην ουρα όλα τα παιδιά του τρέχων κόμβου
		q.push(  new Graph{ Node , UP } );

		q.push(  new Graph{ Node , DOWN } );

		q.push(  new Graph{ Node , LEFT } );

		q.push(  new Graph{ Node , RIGHT } );

    //εάν δεν υπάρχει άλλος κόμβος στην ουρα και δεν βρέθηκε στόχος δεν υπάρχει κανένα μονοπάτι
	} while (!q.empty());

	MEM_CLR();

	return {};
}

bool dfs(const std::vector<std::vector<u8>>& matrix,
	std::pair<i16, i16>position, std::unordered_set<u8 const*>&set ,std::vector<std::pair<i16,i16>>&path)
{
	auto& [x, y] = position;
	/*ελέγχουμε εάν μπορούμε να μετακινηθούμε στην τρέχουσα τοποθεσία δηλαδή εάν η τοποθεσία
	είναι μέσα στα όρια του πινακα ,δεν είναι εμπόδιο και δεν είναι μέσα στο κλειστό σύνολο
	αλλιώς επιστρέφουμε*/
	if ( !AI_ERGASIA_UTILS::ISVALID(position) )
		return false;
	//εφόσον μπορούμε να μετακινηθούμε βάζουμε το κελί στο κλειστό σύνολο και συμπληρώνουμε το μονοπάτι
	set.insert(&matrix[x][y]);
	path.push_back({ x ,y });
	//εάν η τοποθεσία είναι ο στόχος επιστρέφουμε ότι βρέθηκε
	if (matrix[x][y] == 'G')
		return true;

	/*η λύση βρίσκεται είτε στο αριστερό είτε στο δεξί είτε στο κάτω είτε στο πάνω
	υποδέντρο ως αρχικές επιλογές που προφανώς μπορεί να βρίσκεται
	και στα τέσσερα εάν υπάρχουν διασυνδέσεις μεταξύ τους*/
	if (dfs(matrix, { LEFT }, set, path ) ||
		dfs(matrix, { RIGHT }, set, path ) ||
		dfs(matrix, { UP }, set, path ) ||
		dfs(matrix, { DOWN }, set, path ))
		return true;
	/*εάν βρεθούμε σε κόμβο από τον οποιον δεν μπορούμε να συνεχίσουμε και δεν είναι η λύση τότε
	ξεκινάμε να γυρνάμε πίσω αφαιρώντας και τους κόμβους από το μονοπάτι*/
	path.pop_back();
	return false;
}

std::vector<std::pair<i16, i16>> dfs(const std::vector<std::vector<u8>>& matrix, std::pair<i16, i16>position)
{
	//δημιουργείται το κλειστό σύνολο και ο πινακας που θα κρατήσει το μονοπάτι
	std::vector<std::pair< i16, i16 > > path;
	std::unordered_set< u8 const* > set;

	//καλείται η boolean αναδρομική συνάρτηση dfs που θα γεμίσει το path
	return dfs(matrix, position, set, path) ? path : std::vector<std::pair< i16, i16 > >();
}




int main()
{
	
	scope:
		{
			//5 testcases το πρώτο είναι το 5x7 πρότυπο που δόθηκε στην περιγραφή εργασιών και τα αλλα 4 μικρές παραλλαγές
			const auto& test_cases = AI_ERGASIA_UTILS::make_testcases<u8>();
			std::pair<i16, i16> position{ 4,1 };//αρχική θέση

			for (const auto& test : test_cases) {

				                                                  //αλγόριθμος πρώτα σε βάθος
				const std::vector<std::pair<i16, i16>> path_dfs = dfs(test, position);

				                                                 //αλγόριθμος πρώτα σε πλάτος
				const std::vector<std::pair<i16, i16>> path_bfs = bfs(test, position);

				std::cout << "DFS :-\n";
				AI_ERGASIA_UTILS::SHOW_PATH(test, path_dfs);//συνάρτηση που εμφανίζει την διαδρομή

				std::cout << '\n';

				std::cout << "BFS :-\n";
				AI_ERGASIA_UTILS::SHOW_PATH(test, path_bfs);

				std::cout << '\n';
			}


		}


	return 0;
	
}
