package com.hypersale.raglogin.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.nio.file.Path;

@ConfigurationProperties(prefix = "rag")
public class RagProperties {
    private String apiUrl;
    private String internalKey;
    private String documentRoot;
    private String defaultTenant;
    private long uploadMaxBytes;

    public String getApiUrl() { return apiUrl; }
    public void setApiUrl(String apiUrl) { this.apiUrl = apiUrl; }
    public String getInternalKey() { return internalKey; }
    public void setInternalKey(String internalKey) { this.internalKey = internalKey; }
    public Path getDocumentRoot() { return Path.of(documentRoot); }
    public void setDocumentRoot(String documentRoot) { this.documentRoot = documentRoot; }
    public String getDefaultTenant() { return defaultTenant; }
    public void setDefaultTenant(String defaultTenant) { this.defaultTenant = defaultTenant; }
    public long getUploadMaxBytes() { return uploadMaxBytes; }
    public void setUploadMaxBytes(long uploadMaxBytes) { this.uploadMaxBytes = uploadMaxBytes; }
}
