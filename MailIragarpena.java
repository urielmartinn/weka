package proiektua;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

/**
 * Fitxategi anitzetako testuak iragartzeko tresna nagusia.
 * <p>
 * Programa honek aurrez entrenatutako Weka modelo bat kargatzen du eta
 * emandako bi karpetatan dauden testu-fitxategi guztiak sailkatzen ditu.
 * Iragarpenak "iragarpenak.txt" fitxategian gordetzen dira, lerro bakoitzean
 * etiketa eta fitxategi-izena agertzen direla.
 * </p>
 * 
 */
public class MailIragarpena {

    /**
     * Programaren sarrera-puntua.
     * <p>
     * Hiru argumentu espero ditu: ham karpetaren bidea, spam karpetaren bidea
     * eta aurrez gordetako modeloaren fitxategi-bidea.
     * </p>
     * 
     * @param args Komando-lerroko argumentuak:
     *             args[0] = ham karpetaren path-a,
     *             args[1] = spam karpetaren path-a,
     *             args[2] = modeloa (.model fitxategia) gordeta dagoen path-a
     */
    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("Erabilera: java MailIragarpena <ham_direktorioa> <spam_direktorioa> <modeloa.model>");
            System.exit(1);
        }

        String hamDirektorioa = args[0];
        String spamDirektorioa = args[1];
        String modeloIbilbide = args[2];
        String irteera = "iragarpenak.txt";

        try {
            System.out.println("Modeloa kargatzen...");
            Classifier modeloa = (Classifier) SerializationHelper.read(modeloIbilbide);
            Instances dataEgitura = sortuDataEgitura();

            try (BufferedWriter idazlea = new BufferedWriter(new FileWriter(irteera))) {
                direktorioProzesapena(new File(hamDirektorioa), modeloa, dataEgitura, idazlea);
                direktorioProzesapena(new File(spamDirektorioa), modeloa, dataEgitura, idazlea);
            }

            System.out.println("Iragazpenak ateratzen: " + irteera);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Karpeta bateko testu-fitxategi guztiak prozesatzen ditu.
     * <p>
     * Karpeta barruko .txt fitxategi bakoitzaren edukia irakurri, garbitu eta
     * emandako modeloarekin sailkatzen du. Emaitza idazlearen bidez irteera-fitxategira
     * idazten da.
     * </p>
     *
     * @param direktorio Prozesatu beharreko karpeta (ham edo spam)
     * @param modeloa  Aurrez entrenatutako Weka sailkatzailea
     * @param ds     Erabilitako datu-egitura (atributuak eta klasea)
     * @param idazlea Irteera-fitxategira idazteko buffer-ak
     * @throws Exception Irakurketa, garbiketa edo sailkapenean errorea gertatuz gero
     */
    private static void direktorioProzesapena(File direktorio, Classifier modeloa, Instances ds, BufferedWriter idazlea) throws Exception {
        if (!direktorio.exists() || !direktorio.isDirectory()) return;
        File[] fitxategi = direktorio.listFiles((d, izena) -> izena.endsWith(".txt"));
        if (fitxategi == null) return;

        for (File f : fitxategi) {
            String testua = testuaGarbitu(fitxategiaIrakurri(f));
            
            DenseInstance inst = new DenseInstance(2);
            inst.setDataset(ds);
            inst.setValue(0, testua);
            inst.setMissing(1);

            double pred = modeloa.classifyInstance(inst);
            String iragarpena = ds.classAttribute().value((int) pred);

            idazlea.write(iragarpena + ", \"" + f.getName() + "\"\n");
        }
    }

    /**
     * Sailkatzaileak erabiltzeko datu-egitura hutsa sortzen du.
     * <p>
     * Bi atributu ditu: "text" (String motakoa, testu gordina gordetzeko) eta
     * "class" (Nominal motakoa, "ham" eta "spam" balio posibleekin). Klasea
     * bigarren atributua (indizea 1) dela ezartzen du.
     * </p>
     *
     * @return Wekako Instances hutsa, atributuak eta klasea definituak
     */
    private static Instances sortuDataEgitura() {
        ArrayList<Attribute> atributuak = new ArrayList<>();
        atributuak.add(new Attribute("testua", (ArrayList<String>) null));
        ArrayList<String> klaseBalioa = new ArrayList<>();
        klaseBalioa.add("ham");
        klaseBalioa.add("spam");
        atributuak.add(new Attribute("klasea", klaseBalioa));
        Instances ds = new Instances("Inferentzia", atributuak, 0);
        ds.setClassIndex(1);
        return ds;
    }

    /**
     * Fitxategi baten edukia irakurtzen du eta kate bakarrean itzultzen du.
     * <p>
     * Fitxategiko lerro bakoitza zuriune batez bereizita lotzen du, ondorengo
     * prozesamendua errazteko.
     * </p>
     *
     * @param f Irakurri beharreko fitxategia
     * @return Fitxategiaren testu-edukia, lerroak zuriunez bereizita
     * @throws Exception Fitxategia irakurtzean errore bat gertatuz gero
     */
    private static String fitxategiaIrakurri(File f) throws Exception {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String lerroa;
            while ((lerroa = br.readLine()) != null) sb.append(lerroa).append(" ");
        }
        return sb.toString();
    }

    /**
     * Testua garbitzen du, sailkatzailearen errendimendua hobetzeko.
     * <p>
     * Honako eragiketak egiten ditu:
     * <ul>
     *   <li>Posta elektroniko helbideak kendu</li>
     *   <li>URLak kendu</li>
     *   <li>Zenbaki osoak kendu</li>
     *   <li>Letrak ez diren karaktereak kendu (apostrofoak salbu)</li>
     *   <li>Letra xehera bihurtu</li>
     *   <li>Zuriune anitzak kendu</li>
     * </ul>
     * </p>
     *
     * @param text Garbitu beharreko testu gordina
     * @return Garbitutako testua, sailkatzeko prest
     */
    private static String testuaGarbitu(String text) {
        return text.replaceAll("\\S+@\\S+\\.\\S+", " ").replaceAll("(http|https)://\\S+", " ")
                   .replaceAll("\\b\\d+\\b", " ").replaceAll("[^a-zA-Záéíóúñü' ]", " ")
                   .toLowerCase().replaceAll("\\s+", " ").trim();
    }
}
