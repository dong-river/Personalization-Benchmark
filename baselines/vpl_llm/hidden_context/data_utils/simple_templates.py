bird_sentences = [
    "Birds chirp melodiously at the break of dawn.",
    "Birds of prey have keen eyesight for hunting.",
    "Birds migrate to warmer regions during winter.",
    "Birds build intricate nests using twigs and leaves.",
    "Birds exhibit a diverse range of colors and plumage.",
    "Birds communicate through various calls and songs.",
    "Birds have hollow bones which aid in flight.",
    "Birds navigate using the stars and Earth's magnetic field.",
    "Birds' feathers provide insulation and aid in flight.",
    "Birds adapt to different environments for survival.",
    "Birds of paradise have elaborate courtship displays.",
    "Birds use their beaks for feeding and grooming.",
    "Birds flock together for safety and socialization.",
    "Birds molt old feathers to make way for new ones.",
    "Birds play a crucial role in pollination.",
    "Birds like the ostrich are flightless but swift runners.",
    "Birds' songs vary depending on species and region.",
    "Birds mimic human speech and sounds.",
    "Birds exhibit territorial behavior during breeding season.",
    "Birds' eggs come in various sizes and colors.",
    "Birds preen their feathers to keep them clean and waterproof.",
    "Birds of prey hunt smaller animals for food.",
    "Birds use their wings for balance and stability.",
    "Birds migrate thousands of miles during migration.",
    "Birds like the penguin are adapted to life in icy waters.",
    "Birds' nests are often hidden for protection.",
    "Birds have different types of feet for various purposes.",
    "Birds possess remarkable intelligence and problem-solving skills.",
    "Birds' migration patterns are influenced by weather conditions.",
    "Birds huddle together to conserve body heat.",
    "Birds are classified into different orders and families.",
    "Birds' eyesight is sharper than that of humans.",
    "Birds' wingspans vary greatly among species.",
    "Birds roost in trees, cliffs, and buildings.",
    "Birds use vocalizations to establish dominance.",
    "Birds' beaks are adapted to their diet.",
    "Birds' feathers are made of keratin, like human hair and nails.",
    "Birds navigate using landmarks and celestial cues.",
    "Birds like the albatross spend years at sea.",
    "Birds' mating rituals can be elaborate and colorful.",
    "Birds of prey have strong talons for catching prey.",
    "Birds adapt to urban environments for nesting.",
    "Birds migrate to find abundant food sources.",
    "Birds of paradise have elaborate plumage for courtship.",
    "Birds' nests are lined with soft materials for comfort.",
    "Birds like the hummingbird can hover in mid-air.",
    "Birds' songs serve to attract mates and defend territory.",
    "Birds form intricate social structures within flocks.",
    "Birds exhibit different foraging techniques depending on their diet.",
    "Birds' wings allow them to glide effortlessly.",
    "Birds build nests using a variety of materials.",
    "Birds' feathers are crucial for regulating body temperature.",
    "Birds like the eagle have exceptional eyesight.",
    "Birds mark territories with vocalizations and displays.",
    "Birds migrate along established routes called flyways.",
    "Birds' feet are adapted to perching and grasping.",
    "Birds of prey hunt with precision and speed.",
    "Birds adapt their behavior to changes in their environment.",
    "Birds like the flamingo filter-feed in shallow waters.",
    "Birds' calls vary in pitch, tone, and rhythm.",
    "Birds migrate to breeding grounds during spring.",
    "Birds' beaks are adapted to their diet and feeding habits.",
    "Birds' nests vary in size and complexity.",
    "Birds rely on instinct and experience for navigation.",
    "Birds like the swallow undertake long migrations.",
    "Birds exhibit elaborate courtship displays to attract mates.",
    "Birds' eggs are incubated until they hatch.",
    "Birds migrate to avoid harsh winter conditions.",
    "Birds adapt to urban environments by scavenging for food.",
    "Birds' feathers are oiled to repel water.",
    "Birds mark their territories with scent and vocalizations.",
    "Birds display intricate mating rituals to attract partners.",
    "Birds' wings enable them to soar effortlessly.",
    "Birds adapt their behavior to changes in the environment.",
    "Birds build nests in trees, shrubs, and cliffs.",
    "Birds of prey hunt with precision and agility.",
    "Birds' nests are lined with soft materials for insulation.",
    "Birds migrate to warmer climates during winter.",
    "Birds communicate through calls, songs, and displays.",
    "Birds use their beaks for feeding, grooming, and defense.",
    "Birds' feathers provide insulation and aid in flight.",
    "Birds exhibit complex social behaviors within flocks.",
    "Birds adapt to different habitats for survival."
]

cat_sentences = [
    "Cats are mysterious creatures.",
    "Cats possess an independent nature.",
    "Cats enjoy lounging in sunny spots.",
    "Cats have retractable claws.",
    "Cats communicate through meows.",
    "Cats groom themselves meticulously.",
    "Cats can see in low light.",
    "Cats exhibit playful behavior.",
    "Cats are skilled hunters.",
    "Cats have a keen sense of balance.",
    "Cats enjoy chasing toys.",
    "Cats are known for their agility.",
    "Cats often nap throughout the day.",
    "Cats have unique personalities.",
    "Cats knead with their paws.",
    "Cats are territorial animals.",
    "Cats can jump several times their height.",
    "Cats dislike water in general.",
    "Cats have a strong sense of smell.",
    "Cats are crepuscular creatures.",
    "Cats purr when content.",
    "Cats are obligate carnivores.",
    "Cats have excellent hearing.",
    "Cats mark their territory with scent.",
    "Cats have specialized whiskers.",
    "Cats can be trained through positive reinforcement.",
    "Cats are curious by nature.",
    "Cats form strong bonds with their owners.",
    "Cats are skilled climbers.",
    "Cats have a flexible spine.",
    "Cats have a preference for routine.",
    "Cats have a grooming ritual after meals.",
    "Cats use their tails for balance.",
    "Cats have a hierarchy within colonies.",
    "Cats can recognize their names.",
    "Cats exhibit hunting behavior through stalking.",
    "Cats enjoy interactive playtime.",
    "Cats have different vocalizations for various needs.",
    "Cats can sense changes in the weather.",
    "Cats have an acute sense of taste.",
    "Cats show affection through headbutting.",
    "Cats have a natural instinct to hunt rodents.",
    "Cats exhibit territorial spraying behavior.",
    "Cats form social groups with other cats.",
    "Cats can sleep up to 16 hours a day.",
    "Cats are crepuscular hunters.",
    "Cats have an excellent sense of time.",
    "Cats sharpen their claws on scratching posts.",
    "Cats can experience stress from changes in their environment.",
    "Cats have a preference for certain textures.",
    "Cats communicate through body language.",
    "Cats enjoy high perches for observation.",
    "Cats are obligate carnivores, requiring meat in their diet.",
    "Cats can have litters of kittens multiple times a year.",
    "Cats have a strong maternal instinct.",
    "Cats have a hierarchy within multi-cat households.",
    "Cats are prone to hairballs from grooming.",
    "Cats have a sensitive digestive system.",
    "Cats groom other cats in their social group.",
    "Cats have specialized taste receptors.",
    "Cats enjoy hiding in small spaces.",
    "Cats have a unique grooming technique.",
    "Cats have a variety of coat patterns and colors.",
    "Cats have a third eyelid for protection.",
    "Cats display affection through slow blinking.",
    "Cats exhibit whisker fatigue if they touch narrow spaces.",
    "Cats enjoy observing prey from a hidden vantage point.",
    "Cats have a preference for fresh water sources.",
    "Cats can suffer from separation anxiety.",
    "Cats have an instinctual fear of unfamiliar objects.",
    "Cats enjoy toys that mimic prey.",
    "Cats exhibit kneading behavior when relaxed.",
    "Cats can develop allergies to certain foods.",
    "Cats have a sensitive sense of touch in their whiskers.",
    "Cats communicate through scent marking.",
    "Cats have a strong dislike for citrus scents.",
    "Cats can become stressed by changes in routine.",
    "Cats have a preferred sleeping position.",
    "Cats groom to regulate body temperature.",
    "Cats enjoy interactive feeding toys.",
    "Cats have a territorial response to other animals.",
    "Cats have specialized muscles for purring.",
    "Cats have a preferred scratching substrate.",
    "Cats can be trained to walk on a leash.",
    "Cats enjoy sunbathing near windows.",
    "Cats have a preference for certain types of litter.",
    "Cats communicate through vocalizations.",
    "Cats have a natural instinct to bury waste.",
    "Cats are crepuscular hunters by nature.",
    "Cats display affection through grooming rituals.",
    "Cats have a preference for routines in feeding times.",
    "Cats enjoy exploring new environments.",
    "Cats exhibit kneading behavior on soft surfaces.",
    "Cats have a varied vocal range for communication.",
    "Cats have a strong instinct to hunt small prey."
]

dog_sentences = [
    "Dogs are loyal companions.",
    "Dogs come in all shapes and sizes.",
    "Dogs have a keen sense of smell.",
    "Dogs enjoy playing fetch.",
    "Dogs love belly rubs.",
    "Dogs are known as man's best friend.",
    "Dogs make great therapy animals.",
    "Dogs are incredibly intelligent.",
    "Dogs can be trained to perform various tasks.",
    "Dogs need regular exercise.",
    "Dogs enjoy exploring the outdoors.",
    "Dogs have an innate sense of curiosity.",
    "Dogs communicate through body language.",
    "Dogs love treats.",
    "Dogs are pack animals by nature.",
    "Dogs have a strong sense of hierarchy.",
    "Dogs are capable of forming deep bonds with humans.",
    "Dogs provide emotional support to their owners.",
    "Dogs have a natural instinct to protect their families.",
    "Dogs have been domesticated for thousands of years.",
    "Dogs can learn a wide range of commands.",
    "Dogs have an excellent sense of hearing.",
    "Dogs have been used for hunting since ancient times.",
    "Dogs have a playful demeanor.",
    "Dogs are highly adaptable animals.",
    "Dogs have a variety of coat colors and patterns.",
    "Dogs are social animals that thrive on companionship.",
    "Dogs enjoy cuddling with their owners.",
    "Dogs are capable of expressing affection.",
    "Dogs can be trained for search and rescue missions.",
    "Dogs have a strong sense of territory.",
    "Dogs are skilled at interpreting human emotions.",
    "Dogs are often used in police work.",
    "Dogs enjoy participating in agility courses.",
    "Dogs have a remarkable ability to learn new things.",
    "Dogs are known for their unconditional love.",
    "Dogs are descendants of wolves.",
    "Dogs have a natural inclination to chase after moving objects.",
    "Dogs require proper grooming to stay healthy.",
    "Dogs have a unique set of vocalizations.",
    "Dogs have a powerful sense of taste.",
    "Dogs are adept at understanding routines.",
    "Dogs have been depicted in art throughout history.",
    "Dogs are capable of forming friendships with other animals.",
    "Dogs enjoy being part of a family unit.",
    "Dogs can detect changes in human behavior.",
    "Dogs have an extraordinary sense of balance.",
    "Dogs are sensitive to changes in their environment.",
    "Dogs have been trained to assist people with disabilities.",
    "Dogs enjoy playing with toys.",
    "Dogs have an innate sense of direction.",
    "Dogs are known to be protective of children.",
    "Dogs have a natural instinct to dig.",
    "Dogs are skilled at navigating through various terrains.",
    "Dogs enjoy receiving praise from their owners.",
    "Dogs have a strong sense of smell that can detect illness.",
    "Dogs thrive on routine and structure.",
    "Dogs enjoy sunbathing.",
    "Dogs have a playful rivalry with cats.",
    "Dogs require socialization from an early age.",
    "Dogs have a keen sense of time.",
    "Dogs are known to comfort people in distress.",
    "Dogs have a natural affinity for water.",
    "Dogs are capable of learning through observation.",
    "Dogs have been used in therapy for mental health conditions.",
    "Dogs enjoy exploring new scents.",
    "Dogs have a calming presence.",
    "Dogs are known for their ability to sense danger.",
    "Dogs enjoy being praised for good behavior.",
    "Dogs have a strong sense of empathy.",
    "Dogs are skilled at interpreting human gestures.",
    "Dogs have a natural inclination to mark their territory.",
    "Dogs enjoy sleeping in comfortable spots.",
    "Dogs have a playful nature that lasts into old age.",
    "Dogs are excellent at following scent trails.",
    "Dogs have been trained for military purposes.",
    "Dogs are often featured in movies and television shows.",
    "Dogs have a unique way of greeting each other.",
    "Dogs have been companions to humans for millennia.",
    "Dogs enjoy being part of outdoor activities.",
    "Dogs have been known to rescue people in distress.",
    "Dogs have a strong bond with their owners.",
    "Dogs have a calming effect on people.",
    "Dogs have a unique personality.",
    "Dogs have a variety of barks for different situations.",
    "Dogs enjoy exploring their surroundings.",
    "Dogs are known for their sense of playfulness.",
    "Dogs have been used in various forms of work throughout history.",
    "Dogs have a strong sense of loyalty to their families.",
    "Dogs are adept at learning from positive reinforcement.",
    "Dogs have been known to detect natural disasters before they occur.",
    "Dogs have a strong prey drive.",
    "Dogs enjoy being praised and rewarded for their efforts."
]

rabbit_sentences = [
    "Rabbits hop around in fields.",
    "Rabbits have soft fur.",
    "Rabbits eat carrots.",
    "Rabbits reproduce quickly.",
    "Rabbits are social animals.",
    "Rabbits have long ears.",
    "Rabbits twitch their noses.",
    "Rabbits dig burrows.",
    "Rabbits love to play.",
    "Rabbits can be pets.",
    "Rabbits come in many colors.",
    "Rabbits thump their feet.",
    "Rabbits are herbivores.",
    "Rabbits have large families.",
    "Rabbits groom themselves.",
    "Rabbits are prey animals.",
    "Rabbits are agile.",
    "Rabbits are prolific breeders.",
    "Rabbits have a keen sense of smell.",
    "Rabbits are crepuscular.",
    "Rabbits have powerful hind legs.",
    "Rabbits nibble on grass.",
    "Rabbits are cute.",
    "Rabbits have whiskers.",
    "Rabbits communicate through body language.",
    "Rabbits enjoy hiding in tunnels.",
    "Rabbits are fast runners.",
    "Rabbits are nocturnal.",
    "Rabbits have a natural curiosity.",
    "Rabbits have powerful hind legs.",
    "Rabbits need space to roam.",
    "Rabbits have a varied diet.",
    "Rabbits can be trained.",
    "Rabbits are associated with fertility.",
    "Rabbits have a short gestation period.",
    "Rabbits have a lifespan of 8-12 years.",
    "Rabbits enjoy companionship.",
    "Rabbits are quiet animals.",
    "Rabbits are known for their reproductive rate.",
    "Rabbits have a gentle disposition.",
    "Rabbits thump to communicate danger.",
    "Rabbits have a hierarchy within groups.",
    "Rabbits enjoy toys.",
    "Rabbits have strong teeth.",
    "Rabbits are territorial.",
    "Rabbits groom each other.",
    "Rabbits have a complex digestive system.",
    "Rabbits can be litter-trained.",
    "Rabbits binky when happy.",
    "Rabbits have a strong maternal instinct.",
    "Rabbits have a 360-degree field of vision.",
    "Rabbits are prolific diggers.",
    "Rabbits can be affectionate pets.",
    "Rabbits have a unique digestive process called cecotrophy.",
    "Rabbits have a fear of loud noises.",
    "Rabbits can suffer from heatstroke.",
    "Rabbits have a natural inclination to chew.",
    "Rabbits enjoy exploring new environments.",
    "Rabbits have a strong sense of territory.",
    "Rabbits are often depicted in folklore.",
    "Rabbits are used in scientific research.",
    "Rabbits are susceptible to parasites.",
    "Rabbits are good swimmers.",
    "Rabbits have a sensitive respiratory system.",
    "Rabbits have a hutch as their shelter.",
    "Rabbits have a strong sense of balance.",
    "Rabbits are clean animals.",
    "Rabbits can communicate with a range of vocalizations.",
    "Rabbits are known for their fertility.",
    "Rabbits enjoy being petted.",
    "Rabbits need hay for proper digestion.",
    "Rabbits are social creatures.",
    "Rabbits can suffer from loneliness.",
    "Rabbits have a natural instinct to burrow.",
    "Rabbits enjoy fresh vegetables.",
    "Rabbits have a strong maternal bond.",
    "Rabbits can be trained to use a litter box.",
    "Rabbits are crepuscular, meaning they are most active at dawn and dusk.",
    "Rabbits have a delicate skeletal structure.",
    "Rabbits have long been associated with luck.",
    "Rabbits can be territorial over their food.",
    "Rabbits have a complex system of communication.",
    "Rabbits need regular grooming to prevent matting.",
    "Rabbits have a natural curiosity about their environment.",
    "Rabbits enjoy companionship with other rabbits.",
    "Rabbits are very adaptable animals.",
    "Rabbits have a strong sense of smell to detect predators.",
    "Rabbits have a unique digestive system that requires a high-fiber diet.",
    "Rabbits have been domesticated for over 2,000 years.",
    "Rabbits have a keen sense of hearing to detect predators."
]
