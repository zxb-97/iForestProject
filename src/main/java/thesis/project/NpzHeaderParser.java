package thesis.project;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class NpzHeaderParser {
    static int headerLength;
    static byte[] magic;
    static int version;

    public static Map<String,String> npzHeaderReader(InputStream is) throws IOException {
        try(BufferedInputStream bufferedInputStream = new BufferedInputStream(is)){
            // Read magic number
            magic = new byte[6];
            bufferedInputStream.readNBytes(magic,0, magic.length);
            //System.out.println("Magic number = " + new String(magic, StandardCharsets.US_ASCII));

            // Read version number
            byte[] versionBytes = new byte[2];
            bufferedInputStream.read(versionBytes);
            // version is a little-endian unsigned short,
            // java represents stuff in signed big-endian which is reverse order
            // 0xFF masks the sign (ie ignore it), then shift left by 8 bits , then combine the 2 bytes with or
            version = ((versionBytes[1] & 0xFF << 8) | (versionBytes[0] & 0xFF));
            //System.out.println("version = "+ version);

            byte[] headerLenBytes = new byte[2];
            bufferedInputStream.read(headerLenBytes);
            headerLength = ((headerLenBytes[1] & 0xFF << 8) | (headerLenBytes[0] & 0xFF));
            //System.out.println("header length = " + headerLength);

            byte[] headerData = new byte[headerLength];
            bufferedInputStream.read(headerData);
            String headString = new String(headerData, StandardCharsets.US_ASCII).trim();
            //System.out.println(headString);

            Map<String, String> map = parseHeaderString(headString);
            return map;
        }

    }

    private static Map<String, String> parseHeaderString(String headString) {
        Map<String, String> map = new HashMap<>();
        String headNoBraces = "";

        // Remove curly braces of python dict
        if (headString.startsWith("{") && headString.endsWith("}")) {
            headNoBraces = headString.substring(1, headString.length() - 1);
        }

        Pattern pattern = Pattern.compile("([\"']?)([a-zA-Z0-9_]+)\\1\\s*:\\s*('.*?'|\".*?\"|\\(.*?\\)|\\[.*?\\]|\\bTrue\\b|\\bFalse\\b|\\d+\\.?\\d*)");
        Matcher matcher = pattern.matcher(headNoBraces);

        while (matcher.find()) {
            String key = matcher.group(2);
            String value = matcher.group(3).trim(); // remove padding
            map.put(key, value);
        }
        return map;
    }

    static int[] getShapeX(Map<String, String> metadata) {
        int[] shape = new int[2];

        String shapeString = metadata.get("shape");
        String[] parts = shapeString.replaceAll("[()]", "").split(",\\s*");
        int rows = Integer.parseInt(parts[0]);
        int cols = Integer.parseInt(parts[1]);

        shape[0] = rows;
        shape[1] = cols;
        return shape;
    }

    static int getShapeY(Map<String, String> metadata) {
        String shape = metadata.get("shape");
        String[] parts = shape.replaceAll("[()]", "").split(",\\s*");
        int rows = Integer.parseInt(parts[0]);
        return rows;
    }
}
