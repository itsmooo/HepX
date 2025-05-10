"use client"
import Header from "../common/header"
import Footer from "../common/footer"
import { HelpCircle } from "lucide-react"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"

const FaqPage = () => {
  const faqCategories = [
    {
      category: "Ku saabsan Cagaarshowga",
      questions: [
        {
          question: "Waa maxay cagaarshowga?",
          answer:
            "Cagaarshowgu waa bararka beerka. Waxaa sababi kara fayruusyo, khamri, daawooyin, ama waxyaabo kale oo sunta ah. Noocyada ugu badan waa cagaarshowga fayruuska (A, B, C, D, iyo E), kuwaas oo ay sababaan fayruusyo kala duwan oo saameynaya beerka.",
        },
        {
          question: "Maxaa farqi u dhexeeya cagaarshowga A, B, iyo C?",
          answer:
            "Cagaarshowga A waxaa caadi ahaan lagu gudbiyaa cunto ama biyo wasakhaysan waxayna keentaa infekshan xad. Cagaarshowga B waxaa lagu gudbiyaa dhiig iyo dheecaano jirka waxayna noqon kartaa mid xad ah ama joogto ah. Cagaarshowga C waxaa inta badan lagu gudbiyaa xiriirka dhiigga waxayna inta badan noqotaa mid joogto ah, taas oo laga yaabo inay keento dhaawac daran oo beerka ah muddo ka dib.",
        },
        {
          question: "Miyaa la daweyn karaa cagaarshowga?",
          answer:
            "Waxay ku xiran tahay nooca. Cagaarshowga A waxaa caadi ahaan iskii u bogsadaa. Cagaarshowga B waxaa lagu xakameyn karaa dawo haddii ay joogto tahay, in kasta oo aysan jirin daaweyn buuxda. Cagaarshowga C hadda waa la daweyn karaa inta badan kiisaska daawooyin lidka ah fayruuska oo toos ah oo la qaato 8-12 toddobaad.",
        },
      ],
    },
    {
      category: "Astaamaha & Baaritaanka",
      questions: [
        {
          question: "Waa maxay astaamaha caadiga ah ee cagaarshowga?",
          answer:
            "Astaamaha caadiga ah waxaa ka mid ah daal, jaundice (jaalloonaanta maqaarka iyo indhaha), xanuunka caloosha, luminta rabitaanka, lalabbo, matag, kaadi madow, iyo saxaro midab cad. Si kastaba ha ahaatee, dad badan oo qaba cagaarshowga, gaar ahaan marxaladaha hore, ma yeelan karaan wax astaamo ah.",
        },
        {
          question: "Sidee loo ogaadaa cagaarshowga?",
          answer:
            "Baaritaanka caadi ahaan waxaa ka mid ah baaritaannada dhiigga si loo hubiyo shaqada beerka oo loo ogaado unugyada difaaca jirka ama antigen-ka fayruuska. Baaritaanno dheeraad ah waxaa ka mid noqon kara ultrasound, fibroscan, ama biopsy beerka si loo qiimeeyo dhaawaca beerka.",
        },
        {
          question: "Intee in le'eg ayuu sax yahay hubiyaha astaamaha HepaPredict?",
          answer:
            "HepaPredict wuxuu isticmaalaa algorithm-ka barashada mashiinka ee lagu tababaray xogta caafimaadka si loo bixiyo tilmaan ku saabsan cagaarshowga suurtagalka ah oo ku saleysan astaamaha. In kasta oo ay noqon karto tallaabo hore oo waxtar leh, haddana ma aha qalab baaritaan mana bedeli karto talo caafimaad oo xirfad leh iyo la-tashi bixiyeyaasha daryeelka caafimaadka.",
        },
      ],
    },
    {
      category: "Ka hortagga & Daaweynta",
      questions: [
        {
          question: "Sideen uga hortagi karaa inaan qaado cagaarshowga?",
          answer:
            "Habka ka hortagga waxay ku kala duwan yihiin nooca: Cagaarshowga A iyo E, ku dhaqan nadaafad wanaagsan oo ka fogow cunto/biyo wasakhaysan. Cagaarshowga B, tallaalka ayaa la heli karaa waxaana uu waxtar leeyahay. Cagaarshowga C, ka fogow wadaagista cirbadaha ama alaabta shakhsi ahaaneed ee laga yaabo inay xiriir la yeeshaan dhiigga. Dhammaan noocyada, ka fogow isticmaalka khamriga xad-dhaafka ah si aad u ilaaliso beerkaaga.",
        },
        {
          question: "Ma jiraan tallaalada cagaarshowga?",
          answer:
            "Haa, waxaa jira tallaalada ammaan ah oo wax ku ool ah ee cagaarshowga A iyo B, laakiin ma jiraan kuwa cagaarshowga C, D, ama E. Tallaalka cagaarshowga B waxaa si joogto ah loogu taliyaa dhammaan dhallaanka iyo carruurta aan la tallaalin iyo dadka waaweyn ee halista ku jira.",
        },
        {
          question: "Waa maxay daaweynta la heli karo ee cagaarshowga?",
          answer:
            "Daaweynta waxay ku xiran tahay nooca iyo darnaan. Cagaarshowga xad ah inta badan waxay u baahan tahay nasasho iyo kormeer. Cagaarshowga joogtada ah ee B waxay u baahan kartaa daawooyin lidka ah fayruuska. Cagaarshowga C waxaa lagu daweeyaa daawooyin lidka ah fayruuska oo toos ah oo leh heerarka bogsashada aad u sarreeya. Daaweynta waa in had iyo jeer ay hagaan xirfadlayaasha daryeelka caafimaadka.",
        },
      ],
    },
    {
      category: "Isticmaalka HepaPredict",
      questions: [
        {
          question: "Sidee u shaqeeyaa qalabka HepaPredict?",
          answer:
            "HepaPredict wuxuu falanqeeyaa astaamaha aad geliso wuxuuna barbar dhigaa qaababka la xiriira noocyada kala duwan ee cagaarshowga. Isticmaalka algorithm-ka barashada mashiinka ee lagu tababaray xogta caafimaadka, waxay bixisaa qiimayn ku saabsan in astaamahaagu laga yaabo inay tilmaamaan cagaarshowga B, cagaarshowga C, ama midkoodna.",
        },
        {
          question: "Xogtayda ma la ilaalinayaa marka aan isticmaalayo HepaPredict?",
          answer:
            "Haa, waxaan si dhab ah uga xoogaynaa arrimaha la xiriira asturnaanta. Xogta astaamaha ee aad geliso waxaa si ammaan ah loo habeeyaa mana lagu kaydiyo si joogto ah mana la wadaago dhinacyada saddexaad. Ma uruurino macluumaadka aqoonsiga shakhsiga ah iyada oo loo marayo qalabka hubinta astaamaha.",
        },
        {
          question: "Maxaan sameeyaa ka dib marka aan helo natiijo ka timid HepaPredict?",
          answer:
            "Iyadoo aan loo eegin natiiijada, haddii aad la kulanto astaamo ku welwel geliya, waa inaad la tashataa bixiye daryeel caafimaad si aad u hesho baaritaan iyo baaritaan saxda ah. HepaPredict waa qalab waxbarasho oo kor u qaada wacyiga, ma aha qalab baaritaan. Baaritaan saxda ah waxay u baahan tahay baaritaan caafimaad iyo qiimayn xirfad leh.",
        },
      ],
    },
  ]

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-grow">
        <section className="py-16 px-6 bg-gradient-to-b from-hepa-lightTeal/30 to-white">
          <div className="max-w-6xl mx-auto text-center">
            <div className="inline-block p-3 bg-white rounded-full mb-6">
              <HelpCircle className="h-8 w-8 text-hepa-teal" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-hepa-navy mb-6">Su'aalaha Inta Badan La Isweydiiyo</h1>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-8">
              Ka hel jawaabaha su'aalaha caadiga ah ee ku saabsan cagaarshowga iyo isticmaalka qalabka HepaPredict.
            </p>
          </div>
        </section>

        <section className="py-16 px-6">
          <div className="max-w-4xl mx-auto">
            {faqCategories.map((category, categoryIndex) => (
              <div
                key={categoryIndex}
                className="mb-12 animate-fade-in"
                style={{ animationDelay: `${categoryIndex * 0.1}s` }}
              >
                <h2 className="text-2xl font-bold text-hepa-navy mb-6">{category.category}</h2>
                <Accordion type="single" collapsible className="border rounded-lg overflow-hidden">
                  {category.questions.map((faq, faqIndex) => (
                    <AccordionItem
                      key={faqIndex}
                      value={`item-${categoryIndex}-${faqIndex}`}
                      className="border-b last:border-b-0"
                    >
                      <AccordionTrigger className="px-6 py-4 text-left text-hepa-navy hover:text-hepa-teal hover:bg-gray-50 transition-all">
                        {faq.question}
                      </AccordionTrigger>
                      <AccordionContent className="px-6 py-4 text-gray-600">{faq.answer}</AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              </div>
            ))}
          </div>
        </section>
      </main>
      <Footer />
    </div>
  )
}

export default FaqPage
