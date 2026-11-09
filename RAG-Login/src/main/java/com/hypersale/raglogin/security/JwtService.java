package com.hypersale.raglogin.security;

import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.ObjectMapper;
import com.hypersale.raglogin.model.AppUser;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Base64;
import java.util.Map;

@Service
public class JwtService {
    private static final Base64.Encoder ENCODER = Base64.getUrlEncoder().withoutPadding();
    private static final Base64.Decoder DECODER = Base64.getUrlDecoder();
    private final ObjectMapper mapper;
    private final byte[] secret;
    private final long tokenHours;

    public JwtService(ObjectMapper mapper,
                      @Value("${rag.auth-secret}") String secret,
                      @Value("${rag.token-hours:8}") long tokenHours) {
        this.mapper = mapper;
        this.secret = secret.getBytes(StandardCharsets.UTF_8);
        this.tokenHours = tokenHours;
    }

    public String issue(AppUser user) {
        try {
            String header = ENCODER.encodeToString(mapper.writeValueAsBytes(Map.of("alg", "HS256", "typ", "JWT")));
            long now = Instant.now().getEpochSecond();
            String payload = ENCODER.encodeToString(mapper.writeValueAsBytes(Map.of(
                    "sub", user.getUsername(), "uid", user.getId(), "role", user.getRole().name(),
                    "tenant", user.getTenantId(), "iat", now, "exp", now + tokenHours * 3600
            )));
            String unsigned = header + "." + payload;
            return unsigned + "." + ENCODER.encodeToString(sign(unsigned));
        } catch (Exception exception) {
            throw new IllegalStateException("无法创建登录令牌", exception);
        }
    }

    public Map<String, Object> verify(String token) {
        try {
            String[] parts = token.split("\\.");
            if (parts.length != 3) return null;
            String unsigned = parts[0] + "." + parts[1];
            if (!java.security.MessageDigest.isEqual(sign(unsigned), DECODER.decode(parts[2]))) return null;
            Map<String, Object> claims = mapper.readValue(DECODER.decode(parts[1]), new TypeReference<>() {});
            Object expiry = claims.get("exp");
            if (!(expiry instanceof Number number) || number.longValue() <= Instant.now().getEpochSecond()) return null;
            if (!"HS256".equals(mapper.readTree(DECODER.decode(parts[0])).path("alg").asText())) return null;
            return claims;
        } catch (Exception exception) {
            return null;
        }
    }

    private byte[] sign(String value) throws Exception {
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(secret, "HmacSHA256"));
        return mac.doFinal(value.getBytes(StandardCharsets.UTF_8));
    }
}
