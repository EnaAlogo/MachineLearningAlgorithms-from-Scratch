#include "lib.cpp"

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