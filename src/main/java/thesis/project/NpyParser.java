package thesis.project;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;


import static thesis.project.NpyHeaderParser.*;

interface ParserX{
    double[][] parse(File path) throws IOException;
}
interface ParserY{
    long[] parse(File path) throws IOException;
}

class xParser implements ParserX{
    public double[][] parse(File path) throws IOException{
        // Getting info stored in header
        Map<String, String> metadata = npyHeaderReader(path);

        int bufferSize = metadata.get("descr").charAt(3) - '0'; // descr = '<f8' or '<i2' etc
        char ndarrayType = metadata.get("descr").charAt(2);

        int offset = 6 + 1 + 1 + 2 + headerLength;

        int rows = getShapeX(metadata)[0];
        int cols = getShapeX(metadata)[1];

        double[][] data = new double[rows][cols];

        // Get the data as a FileInputStream object
        try(FileInputStream fileStream = new FileInputStream(path)){
            // Wrap it into a BufferedInputStream object to use the functionality of this class
            BufferedInputStream fileBuffer = new BufferedInputStream(fileStream);
            // Skip the header to start the read from where the actual data resides
            fileBuffer.skipNBytes(offset);

            byte[] buffer = new byte[bufferSize];
            // Wrap the buffer so that when transforming the bytes into a java type
            // the get method (getDouble(), getLong()..) knows that it has to read the buffer in little endian
            // this is necessary because java types are in big endian
            ByteBuffer byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

            // X.npy are in Fortran order
            for (int i = 0; i < cols; i++){
                for(int j = 0; j < rows; j++){
                    if(fileBuffer.read(buffer) != bufferSize ){
                        throw new IOException("failed to read bytes in row " + j + " column " + i);
                    }
                    // Reset buffer
                    byteBuffer.clear();

                    if(ndarrayType == 'f'){
                        data[j][i] = byteBuffer.getDouble();
                    }
                    if((ndarrayType == 'i') && (bufferSize == 8)){
                        data[j][i] = (double) byteBuffer.getLong();
                    }
                    if((ndarrayType == 'i') && (bufferSize == 4)){
                        data[j][i] = (double) byteBuffer.getInt();
                    }
                    if((ndarrayType == 'i') && (bufferSize == 2)){
                        data[j][i] = (double) byteBuffer.getShort();
                    }
                }
            }
            return data;
        }
    }
}

class yParser implements ParserY{

    public long[] parse(File path) throws IOException {
        Map<String, String> metadata = npyHeaderReader(path);
        int bufferSize = metadata.get("descr").charAt(3) - '0';
        char ndarrayType = metadata.get("descr").charAt(2);

        int offset = 10 + headerLength;
        int rows = getShapeY(metadata);
        long[] data = new long[rows];

        try(FileInputStream fileStream = new FileInputStream(path)){
            BufferedInputStream fileBuffer = new BufferedInputStream(fileStream);
            fileBuffer.skipNBytes(offset);

            byte[] buffer = new byte[bufferSize];

            ByteBuffer byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

            for(int i = 0; i < rows; i++){
                if(fileBuffer.read(buffer) != bufferSize){
                    throw new IOException("failed to read in row: " + i);
                }
                // Reset read position
                byteBuffer.rewind();
                if(ndarrayType == 'i' && bufferSize == 8){
                    data[i] = byteBuffer.getLong();
                }
                if(ndarrayType == 'i' && bufferSize == 4){
                    data[i] = byteBuffer.getInt();
                }
                if(ndarrayType == 'i' && bufferSize == 2){
                    data[i] = byteBuffer.getShort();
                }
            }
            return data;
        }
    }
}

public class NpyParser{
    static ParserX getParserForX() {
        return new xParser();
    }

    static ParserY getParserForY() {
        return new yParser();
    }
}





