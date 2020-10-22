package neural;
import java.util.Arrays;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import javax.imageio.ImageIO;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import robocode.BattleResults;
import robocode.control.*;
import robocode.control.events.*;

public class BattlefieldParameterEvaluator 
{
	final static int MINBATTLEFIELDSIZE = 400;
	final static int MAXBATTLEFIELDSIZE = 800;
	final static double MINGUNCOOLINGRATE = 0.1;
	final static double MAXGUNCOOLINGRATE = 0.5;
	final static int NUMBATTLEFIELDSIZES = 601;
	final static int NUMCOOLINGRATES = 501;
	final static int NUMSAMPLES = 1000;
	final static int NUM_NN_INPUTS = 2;
	final static int NUM_NN_HIDDEN_UNITS = 50;
	final static int NUM_TRAINING_EPOCHS = 100000;
	final static int MAXNUMTURNS = 10000;
	
	static int NdxBattle;
	static double []NumTurns;
	
	
	public static void main(String[] args) {
		double []BattlefieldSize = new double[NUMSAMPLES];
		double []GunCoolingRate = new double[NUMSAMPLES];
		NumTurns = new double[NUMSAMPLES];
		
		Random rng = new Random(15L);
		
		RobocodeEngine.setLogMessagesEnabled(false);
		RobocodeEngine engine = new RobocodeEngine(new java.io.File("C:/Robocode"));
		engine.addBattleListener(new BattleObserver());
		engine.setVisible(true);

		int numberOfRounds = 1;
		long inactivityTime = 100;
		int sentryBorderSize = 50;
		boolean hideEnemyNames = false;

		RobotSpecification[] competingRobots = engine.getLocalRepository("sample.SittingDuck, sample.TrackFire");
		for(NdxBattle = 0; NdxBattle < NUMSAMPLES; NdxBattle++)
		{
			// Choose the battlefield size and gun cooling rate
			BattlefieldSize[NdxBattle] = MINBATTLEFIELDSIZE + (rng.nextDouble() * (MAXBATTLEFIELDSIZE - MINBATTLEFIELDSIZE));
			GunCoolingRate[NdxBattle] = MINGUNCOOLINGRATE + (rng.nextDouble() * (MAXGUNCOOLINGRATE - MINGUNCOOLINGRATE));
			
			System.out.println(BattlefieldSize[NdxBattle]);
			System.out.println(GunCoolingRate[NdxBattle]);

			
			// Create the battlefield
			BattlefieldSpecification battlefield =
			new BattlefieldSpecification((int)BattlefieldSize[NdxBattle], (int)BattlefieldSize[NdxBattle]);
			
			// Set the robot positions		
			int boardSize = (int) Math.floor((BattlefieldSize[NdxBattle] - 32) / 64);
			int numSittingDucks = (int) Math.round(0.0001 * BattlefieldSize[NdxBattle] * BattlefieldSize[NdxBattle]);
			RobotSetup[] robotSetups = new RobotSetup[numSittingDucks + 1];
			RobotSpecification[] existingRobots = new RobotSpecification[numSittingDucks + 1];
			
			int row = 0;
			int column = 0;
			for(int i = 0; i < numSittingDucks; i ++)
			{
				existingRobots[i] = competingRobots[0];
				robotSetups[i] = new RobotSetup(32.0 + (column * 64), 32.0 + (row * 64), 0.0);
				
				column++;
				
				if(column % (boardSize) == 0)
				{
					row++;
					column = 0;
				}
			}
			
			existingRobots[numSittingDucks] = competingRobots[1];
			robotSetups[numSittingDucks] = new RobotSetup(32.0 + (boardSize - 1) * 64, 32.0 + (boardSize - 1) * 64, 0.0);
			
			
			// Prepare the battle specification
			BattleSpecification battleSpec = new BattleSpecification(
					battlefield,
					numberOfRounds,
					inactivityTime,
					GunCoolingRate[NdxBattle],
					sentryBorderSize,
					hideEnemyNames,
					existingRobots,
					robotSetups);
			
			// Run our specified battle and let it run till it is over
			engine.runBattle(battleSpec, true); // waits till the battle finishes
		}
		
		// Cleanup our RobocodeEngine
		engine.close();
		System.out.println(Arrays.toString(BattlefieldSize));
		System.out.println(Arrays.toString(GunCoolingRate));
		System.out.println(Arrays.toString(NumTurns));
		
		double [][]RawInputs=new double[NUMSAMPLES][NUM_NN_INPUTS];
		double [][]RawOutputs=new double[NUMSAMPLES][1];
		
		for(int NdxSample = 0; NdxSample < NUMSAMPLES; NdxSample++)
		{
			RawInputs[NdxSample][0] = BattlefieldSize[NdxSample] / MAXBATTLEFIELDSIZE;
			RawInputs[NdxSample][1] = GunCoolingRate[NdxSample] / MAXGUNCOOLINGRATE;
			RawOutputs[NdxSample][0] = NumTurns[NdxSample] / MAXNUMTURNS;
		}
		
		BasicNetwork network = generateNetwork();
			
		BasicNeuralDataSet MyDataSet = new BasicNeuralDataSet(RawInputs, RawOutputs);			
		System.out.println("Training network...");
		final ResilientPropagation train = new ResilientPropagation(network, MyDataSet);
				
		for(int i = 0; i < NUM_TRAINING_EPOCHS; i++)
		{
			train.iteration();
			System.out.println("Epoch #" + i + "Error: " + train.getError());
		}

		train.finishTraining();		
		System.out.println("Training completed.");
		
		
		System.out.println("Testing network...");		
		printImg(GunCoolingRate, BattlefieldSize, network);
		
		Encog.getInstance().shutdown();
		System.exit(0);
	 
	}
	
	public static BasicNetwork generateNetwork()
	{
		final BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(NUM_NN_INPUTS));
		network.addLayer(new BasicLayer(NUM_NN_HIDDEN_UNITS));
		network.addLayer(new BasicLayer(1));
		network.getStructure().finalizeStructure();
		network.reset();
		return network;
	}
	

	public static void printImg(double GunCoolingRate[], double BattlefieldSize[], BasicNetwork network)
	{
		// Generate test samples to build an output image
		int []OutputRGBint = new int[NUMBATTLEFIELDSIZES * NUMCOOLINGRATES];
		Color MyColor;
		double MyValue = 0;
		double [][]MyTestData = new double [NUMBATTLEFIELDSIZES * NUMCOOLINGRATES][NUM_NN_INPUTS];
		
		for(int NdxBattleSize=0; NdxBattleSize < NUMBATTLEFIELDSIZES; NdxBattleSize++)
		{
			for(int NdxCooling = 0; NdxCooling < NUMCOOLINGRATES; NdxCooling++)
			{
				MyTestData[NdxCooling + NdxBattleSize * NUMCOOLINGRATES][0] = 0.1 + 0.9 * ((double)NdxBattleSize) / NUMBATTLEFIELDSIZES;
				MyTestData[NdxCooling + NdxBattleSize * NUMCOOLINGRATES][1] = 0.1 + 0.9 * ((double)NdxCooling) / NUMCOOLINGRATES;
			}
		}
		
		for(int NdxBattleSize=0;NdxBattleSize<NUMBATTLEFIELDSIZES;NdxBattleSize++)
		{
			for(int NdxCooling = 0; NdxCooling < NUMCOOLINGRATES; NdxCooling++)
			{
				
				double MyResult = network.compute(new BasicMLData(MyTestData[NdxCooling + NdxBattleSize * NUMCOOLINGRATES])).getData(0);
								
				MyValue = ClipColor(MyResult);
				MyColor = new Color((float)MyValue, (float)MyValue, (float)MyValue);
				OutputRGBint[NdxCooling + NdxBattleSize * NUMCOOLINGRATES] = MyColor.getRGB();
			}
		}
		
		BufferedImage img=new BufferedImage
		(NUMCOOLINGRATES,NUMBATTLEFIELDSIZES,BufferedImage.TYPE_INT_RGB);
		img.setRGB(0, 0, NUMCOOLINGRATES, NUMBATTLEFIELDSIZES,
		OutputRGBint, 0, NUMCOOLINGRATES);
		
		File f=new File("test.png");
		try {
			ImageIO.write(img,"png",f);
		}catch (IOException e) {
			e.printStackTrace();
		}		
		System.out.println("Image generated.");
		System.out.println("Testing completed.");
		
		
		
		
		int []OutputRGBint2 = new int[NUMBATTLEFIELDSIZES * NUMCOOLINGRATES];
		// Plot the training samples
		for(int NdxSample=0;NdxSample<NUMSAMPLES;NdxSample++)
		{
			MyValue=ClipColor(NumTurns[NdxSample]/MAXNUMTURNS);
			MyColor = new Color((float)MyValue, (float)MyValue, (float)MyValue);
			int MyPixelIndex = (int)(Math.round(NUMCOOLINGRATES * (GunCoolingRate[NdxSample] / MAXGUNCOOLINGRATE))
					+ Math.round(NUMBATTLEFIELDSIZES * (BattlefieldSize[NdxSample] / MAXBATTLEFIELDSIZE)) * NUMCOOLINGRATES);
			
			if ((MyPixelIndex >= 0) && (MyPixelIndex < NUMCOOLINGRATES * NUMBATTLEFIELDSIZES))
			{
				OutputRGBint2[MyPixelIndex] = MyColor.getRGB();
			}
		}
		
		BufferedImage img2=new BufferedImage
		(NUMCOOLINGRATES,NUMBATTLEFIELDSIZES,BufferedImage.TYPE_INT_RGB);
		img2.setRGB(0, 0, NUMCOOLINGRATES, NUMBATTLEFIELDSIZES,
		OutputRGBint2, 0, NUMCOOLINGRATES);
		
		File f2=new File("train.png");
		try {
			ImageIO.write(img2,"png",f2);
		}catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("Image generated.");
	}
	
	public static double ClipColor(double Value)
	{
		if(Value<0.0)
			Value=0.0;

		if(Value>1.0)
			Value=1.0;
		
		return Value;
	}
	
	static class BattleObserver extends BattleAdaptor {	
		public void onRoundEnded(RoundEndedEvent e)
		{
			int turns = e.getTurns();
			System.out.println("-- Battle " + NdxBattle + " has ended --");
			System.out.println("Number of turns taken:");	
			System.out.println(turns);
			BattlefieldParameterEvaluator.NumTurns[NdxBattle] = turns;		
		}
		
		public void onBattleCompleted(BattleCompletedEvent e) {	
		}
		
		// Called when the game sends out an information message during the battle
		public void onBattleMessage(BattleMessageEvent e) {
		//System.out.println("Msg> " + e.getMessage());
		}
		
		// Called when the game sends out an error message during the battle
		public void onBattleError(BattleErrorEvent e) {
			System.out.println("Err> " + e.getError());
		}
		
	}
}